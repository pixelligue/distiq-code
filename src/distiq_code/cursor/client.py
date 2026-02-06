"""Cursor API HTTP/2 client with proper protobuf encoding.

Handles communication with Cursor's backend API using ConnectRPC protocol.
Based on reverse-engineered schema from cursor_api_demo.
"""

import base64
import gzip
import hashlib
import platform
import struct
import time
import uuid
from datetime import datetime
from typing import AsyncIterator, Optional

import httpx
from loguru import logger

from distiq_code.cursor.auth import CursorAuth, CursorTokens


class ProtobufEncoder:
    """Manual protobuf encoder (no .proto compilation needed)."""

    @staticmethod
    def encode_varint(value: int) -> bytes:
        """Encode integer as protobuf varint."""
        result = b""
        while value >= 0x80:
            result += bytes([value & 0x7F | 0x80])
            value >>= 7
        result += bytes([value & 0x7F])
        return result

    @staticmethod
    def encode_field(field_num: int, wire_type: int, value) -> bytes:
        """
        Encode protobuf field.

        Args:
            field_num: Field number from .proto schema
            wire_type: 0=varint, 1=64-bit, 2=length-delimited, 5=32-bit
            value: Field value (int, bytes, or str)

        Returns:
            Encoded field bytes
        """
        tag = (field_num << 3) | wire_type
        result = ProtobufEncoder.encode_varint(tag)

        if wire_type == 0:  # Varint
            result += ProtobufEncoder.encode_varint(value)
        elif wire_type == 2:  # Length-delimited (string/bytes/submessage)
            if isinstance(value, str):
                value = value.encode("utf-8")
            result += ProtobufEncoder.encode_varint(len(value)) + value

        return result


class CursorClient:
    """
    HTTP/2 client for Cursor API with proper protobuf encoding.

    Uses ConnectRPC (gRPC-Web variant) with binary protobuf.
    Requires HTTP/2 transport (HTTP/1.1 returns error 464).
    """

    API_BASE = "https://api2.cursor.sh"
    CHAT_ENDPOINT = "/aiserver.v1.ChatService/StreamUnifiedChatWithTools"

    def __init__(self, tokens: Optional[CursorTokens] = None):
        """
        Initialize Cursor client.

        Args:
            tokens: Pre-extracted tokens, or None to auto-extract from database
        """
        self.tokens = tokens or CursorAuth.extract_tokens()

        # HTTP/2 client (required!)
        self.client = httpx.AsyncClient(http2=True, timeout=120.0)

        logger.info(f"Cursor client initialized (email: {self.tokens.email})")

    def _generate_checksum(self) -> str:
        """
        Generate x-cursor-checksum header using Jyh cipher.

        This is a custom obfuscation algorithm used by Cursor for request validation.

        Returns:
            Base64-encoded checksum string
        """
        # Current timestamp in seconds (divided by 1M for some reason)
        timestamp = int(time.time() * 1000 // 1000000)

        # Convert timestamp to 6-byte array (big-endian)
        byte_array = bytearray([
            (timestamp >> 40) & 0xFF,
            (timestamp >> 32) & 0xFF,
            (timestamp >> 24) & 0xFF,
            (timestamp >> 16) & 0xFF,
            (timestamp >> 8) & 0xFF,
            timestamp & 0xFF,
        ])

        # Jyh cipher: XOR with rotating key
        key = 165
        for i in range(6):
            byte_array[i] = ((byte_array[i] ^ key) + (i % 256)) & 0xFF
            key = byte_array[i]  # Key rotates

        # URL-safe base64 encode (custom alphabet)
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        encoded = ""
        for i in range(0, len(byte_array), 3):
            a = byte_array[i]
            b = byte_array[i + 1] if i + 1 < len(byte_array) else 0
            c = byte_array[i + 2] if i + 2 < len(byte_array) else 0
            encoded += alphabet[a >> 2]
            encoded += alphabet[((a & 3) << 4) | (b >> 4)]
            if i + 1 < len(byte_array):
                encoded += alphabet[((b & 15) << 2) | (c >> 6)]
            if i + 2 < len(byte_array):
                encoded += alphabet[c & 63]

        checksum = f"{encoded}{self.tokens.machine_id}"
        return checksum

    def _generate_session_id(self) -> str:
        """
        Generate x-session-id header.

        Uses UUID5 derived from access token.

        Returns:
            Session ID string
        """
        # UUID5 with DNS namespace
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, self.tokens.access_token))

    def _generate_client_key(self) -> str:
        """
        Generate x-client-key header.

        SHA256 hash of access token.

        Returns:
            Client key hex string
        """
        return hashlib.sha256(self.tokens.access_token.encode()).hexdigest()

    def _build_headers(self) -> dict[str, str]:
        """
        Build HTTP headers for Cursor API request.

        Returns:
            Headers dictionary with all required authentication headers
        """
        return {
            "Authorization": f"Bearer {self.tokens.access_token}",
            "Content-Type": "application/connect+proto",
            "Connect-Protocol-Version": "1",
            "User-Agent": "connect-es/1.6.1",
            "Accept-Encoding": "gzip",
            "x-cursor-checksum": self._generate_checksum(),
            "x-session-id": self._generate_session_id(),
            "x-client-key": self._generate_client_key(),
            "x-client-version": "0.47.2",
            "x-cursor-client-type": "vscode",
            "x-machine-id": self.tokens.machine_id,
            "x-amzn-trace-id": f"Root={uuid.uuid4()}",
        }

    def _encode_message(
        self, content: str, role: int, message_id: str, chat_mode_enum: Optional[int] = None
    ) -> bytes:
        """
        Encode Message protobuf.

        Args:
            content: Message text
            role: 1=user, 2=assistant
            message_id: UUID string
            chat_mode_enum: Optional chat mode (1=Ask, only for user messages)

        Returns:
            Encoded Message bytes
        """
        msg = b""

        # string content = 1;
        msg += ProtobufEncoder.encode_field(1, 2, content)

        # int32 role = 2;
        msg += ProtobufEncoder.encode_field(2, 0, role)

        # string messageId = 13;
        msg += ProtobufEncoder.encode_field(13, 2, message_id)

        # int32 chatModeEnum = 47;
        if chat_mode_enum is not None:
            msg += ProtobufEncoder.encode_field(47, 0, chat_mode_enum)

        return msg

    def _encode_instruction(self, instruction_text: str) -> bytes:
        """Encode Instruction protobuf."""
        msg = b""

        # string instruction = 1;
        if instruction_text:
            msg += ProtobufEncoder.encode_field(1, 2, instruction_text)

        return msg

    def _encode_model(self, model_name: str) -> bytes:
        """Encode Model protobuf."""
        msg = b""

        # string name = 1;
        msg += ProtobufEncoder.encode_field(1, 2, model_name)

        # bytes empty = 4;
        msg += ProtobufEncoder.encode_field(4, 2, b"")

        return msg

    def _encode_cursor_setting(self) -> bytes:
        """Encode CursorSetting protobuf."""
        msg = b""

        # string name = 1;
        msg += ProtobufEncoder.encode_field(1, 2, "cursor\\aisettings")

        # bytes unknown3 = 3;
        msg += ProtobufEncoder.encode_field(3, 2, b"")

        # Unknown6 unknown6 = 6;
        unknown6_msg = b""
        unknown6_msg += ProtobufEncoder.encode_field(1, 2, b"")  # bytes unknown1 = 1
        unknown6_msg += ProtobufEncoder.encode_field(2, 2, b"")  # bytes unknown2 = 2
        msg += ProtobufEncoder.encode_field(6, 2, unknown6_msg)

        # int32 unknown8 = 8;
        msg += ProtobufEncoder.encode_field(8, 0, 1)

        # int32 unknown9 = 9;
        msg += ProtobufEncoder.encode_field(9, 0, 1)

        return msg

    def _encode_metadata(self) -> bytes:
        """Encode Metadata protobuf."""
        msg = b""

        # string os = 1;
        system = platform.system().lower()
        if system == "darwin":
            system = "macos"
        msg += ProtobufEncoder.encode_field(1, 2, system)

        # string arch = 2;
        arch = platform.machine().lower()
        if arch == "amd64":
            arch = "x64"
        msg += ProtobufEncoder.encode_field(2, 2, arch)

        # string version = 3;
        msg += ProtobufEncoder.encode_field(3, 2, "0.47.2")

        # string path = 4;
        msg += ProtobufEncoder.encode_field(4, 2, "/usr/bin/cursor")

        # string timestamp = 5;
        msg += ProtobufEncoder.encode_field(5, 2, datetime.now().isoformat())

        return msg

    def _encode_message_id(self, message_id: str, role: int) -> bytes:
        """Encode MessageId protobuf."""
        msg = b""

        # string messageId = 1;
        msg += ProtobufEncoder.encode_field(1, 2, message_id)

        # int32 role = 3;
        msg += ProtobufEncoder.encode_field(3, 0, role)

        return msg

    def _encode_request(self, messages: list[dict], model_name: str) -> bytes:
        """
        Encode Request protobuf (the inner request object).

        Args:
            messages: List of message dicts with 'role' and 'content'
            model_name: Model to use

        Returns:
            Encoded Request bytes
        """
        msg = b""

        # Format messages and collect IDs
        formatted_messages = []
        message_ids = []

        for user_msg in messages:
            if user_msg["role"] == "user":
                msg_id = str(uuid.uuid4())
                formatted_messages.append({
                    "content": user_msg["content"],
                    "role": 1,  # user
                    "messageId": msg_id,
                    "chatModeEnum": 1,  # Ask mode
                })
                message_ids.append({"messageId": msg_id, "role": 1})

        # repeated Message messages = 1;
        for formatted_msg in formatted_messages:
            message_bytes = self._encode_message(
                formatted_msg["content"],
                formatted_msg["role"],
                formatted_msg["messageId"],
                formatted_msg.get("chatModeEnum"),
            )
            msg += ProtobufEncoder.encode_field(1, 2, message_bytes)

        # int32 unknown2 = 2;
        msg += ProtobufEncoder.encode_field(2, 0, 1)

        # Instruction instruction = 3;
        instruction_bytes = self._encode_instruction("")
        msg += ProtobufEncoder.encode_field(3, 2, instruction_bytes)

        # int32 unknown4 = 4;
        msg += ProtobufEncoder.encode_field(4, 0, 1)

        # Model model = 5;
        model_bytes = self._encode_model(model_name)
        msg += ProtobufEncoder.encode_field(5, 2, model_bytes)

        # string webTool = 8;
        msg += ProtobufEncoder.encode_field(8, 2, "")

        # int32 unknown13 = 13;
        msg += ProtobufEncoder.encode_field(13, 0, 1)

        # CursorSetting cursorSetting = 15;
        cursor_setting_bytes = self._encode_cursor_setting()
        msg += ProtobufEncoder.encode_field(15, 2, cursor_setting_bytes)

        # int32 unknown19 = 19;
        msg += ProtobufEncoder.encode_field(19, 0, 1)

        # string conversationId = 23;
        msg += ProtobufEncoder.encode_field(23, 2, str(uuid.uuid4()))

        # Metadata metadata = 26;
        metadata_bytes = self._encode_metadata()
        msg += ProtobufEncoder.encode_field(26, 2, metadata_bytes)

        # int32 unknown27 = 27;
        msg += ProtobufEncoder.encode_field(27, 0, 0)

        # repeated MessageId messageIds = 30;
        for msg_id_data in message_ids:
            message_id_bytes = self._encode_message_id(msg_id_data["messageId"], msg_id_data["role"])
            msg += ProtobufEncoder.encode_field(30, 2, message_id_bytes)

        # int32 largeContext = 35;
        msg += ProtobufEncoder.encode_field(35, 0, 0)

        # int32 unknown38 = 38;
        msg += ProtobufEncoder.encode_field(38, 0, 0)

        # int32 chatModeEnum = 46;
        msg += ProtobufEncoder.encode_field(46, 0, 1)

        # string unknown47 = 47;
        msg += ProtobufEncoder.encode_field(47, 2, "")

        # int32 unknown48 = 48;
        msg += ProtobufEncoder.encode_field(48, 0, 0)

        # int32 unknown49 = 49;
        msg += ProtobufEncoder.encode_field(49, 0, 0)

        # int32 unknown51 = 51;
        msg += ProtobufEncoder.encode_field(51, 0, 0)

        # int32 unknown53 = 53;
        msg += ProtobufEncoder.encode_field(53, 0, 1)

        # string chatMode = 54;
        msg += ProtobufEncoder.encode_field(54, 2, "Ask")

        return msg

    def _encode_stream_unified_chat_request(self, messages: list[dict], model_name: str) -> bytes:
        """
        Encode StreamUnifiedChatWithToolsRequest protobuf.

        Args:
            messages: List of message dicts
            model_name: Model to use

        Returns:
            Encoded StreamUnifiedChatWithToolsRequest bytes
        """
        msg = b""

        # Request request = 1;
        request_bytes = self._encode_request(messages, model_name)
        msg += ProtobufEncoder.encode_field(1, 2, request_bytes)

        return msg

    def _wrap_with_envelope(self, protobuf_payload: bytes, compress: bool = False) -> bytes:
        """
        Wrap protobuf payload in ConnectRPC envelope with optional gzip compression.

        Format: [flags:1][length:4_BE][payload]

        Args:
            protobuf_payload: Binary protobuf message
            compress: Apply gzip compression (used for messages with 3+ turns)

        Returns:
            Enveloped message ready for HTTP/2 transport
        """
        # Apply gzip if requested
        if compress:
            protobuf_payload = gzip.compress(protobuf_payload)
            flags = 0x01  # Compressed
        else:
            flags = 0x00  # Uncompressed

        length = len(protobuf_payload)
        envelope = bytes([flags]) + struct.pack(">I", length) + protobuf_payload

        return envelope

    async def chat(
        self,
        prompt: str,
        model: str = "cursor-small",
        conversation_id: Optional[str] = None,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """
        Send chat request to Cursor API and stream response.

        Args:
            prompt: User message
            model: Model to use (cursor-small, claude-4.5-opus-high, etc.)
            conversation_id: Optional conversation ID (not used in current impl)
            stream: Enable streaming (required for Cursor API)

        Yields:
            Text chunks from assistant response

        Raises:
            httpx.HTTPStatusError: If API request fails
        """
        # Build messages array
        messages = [{"role": "user", "content": prompt}]

        # Encode request
        protobuf_payload = self._encode_stream_unified_chat_request(messages, model)

        # Wrap with envelope (compress if 3+ messages)
        compress = len(messages) >= 3
        enveloped_payload = self._wrap_with_envelope(protobuf_payload, compress=compress)

        # Build headers
        headers = self._build_headers()

        # Send request
        url = f"{self.API_BASE}{self.CHAT_ENDPOINT}"

        logger.info(
            f"Sending Cursor API request (model: {model}, "
            f"len: {len(prompt)} chars, compressed: {compress})"
        )

        async with self.client.stream("POST", url, headers=headers, content=enveloped_payload) as response:
            response.raise_for_status()

            # Parse streaming response
            buffer = bytearray()

            async for chunk in response.aiter_bytes():
                buffer.extend(chunk)

                # Parse frames
                while len(buffer) >= 5:
                    # Frame format: [type:1][length:4_BE][data]
                    frame_type = buffer[0]
                    frame_length = struct.unpack(">I", buffer[1:5])[0]

                    # Check if full frame available
                    if len(buffer) < 5 + frame_length:
                        break

                    # Extract frame data
                    frame_data = buffer[5 : 5 + frame_length]
                    buffer = buffer[5 + frame_length :]

                    # Decode frame
                    text = None

                    if frame_type == 0:
                        # Raw protobuf
                        text = self._extract_text_from_protobuf(frame_data)
                    elif frame_type == 1:
                        # Gzip protobuf
                        decompressed = gzip.decompress(frame_data)
                        text = self._extract_text_from_protobuf(decompressed)
                    elif frame_type == 2:
                        # Raw JSON
                        import json

                        try:
                            data = json.loads(frame_data)
                            text = self._extract_text_from_json(data)
                        except Exception as e:
                            logger.debug(f"Failed to parse JSON frame: {e}")
                    elif frame_type == 3:
                        # Gzip JSON
                        import json

                        try:
                            decompressed = gzip.decompress(frame_data)
                            data = json.loads(decompressed)
                            text = self._extract_text_from_json(data)
                        except Exception as e:
                            logger.debug(f"Failed to parse gzip JSON frame: {e}")
                    else:
                        logger.debug(f"Unknown frame type: {frame_type}")

                    if text:
                        yield text

    def _extract_text_from_protobuf(self, data: bytes) -> Optional[str]:
        """
        Extract text content from protobuf response.

        Simple parser that looks for text field (field 1, wire type 2).

        Args:
            data: Binary protobuf data

        Returns:
            Extracted text or None
        """
        try:
            # Very basic protobuf parser
            # Look for field 1 (tag = 0x0a = field_num=1, wire_type=2)
            idx = 0
            while idx < len(data):
                # Check for field 1 tag
                if data[idx] == 0x0A:
                    # Next byte(s) is length varint
                    idx += 1
                    if idx >= len(data):
                        break

                    # Decode varint length (simplified: assume < 128)
                    length = data[idx]
                    if length >= 128:
                        # Multi-byte varint, skip for now
                        break

                    idx += 1
                    if idx + length > len(data):
                        break

                    # Extract text
                    text_bytes = data[idx : idx + length]
                    text = text_bytes.decode("utf-8", errors="ignore")
                    if text.strip():
                        return text

                idx += 1

        except Exception as e:
            logger.debug(f"Failed to extract text from protobuf: {e}")

        return None

    def _extract_text_from_json(self, data: dict) -> Optional[str]:
        """
        Extract text from JSON response.

        Args:
            data: Parsed JSON dict

        Returns:
            Extracted text or None
        """
        if isinstance(data, dict):
            # Check common text fields
            for field in ["text", "content", "message", "delta"]:
                if field in data:
                    value = data[field]
                    if isinstance(value, str) and value.strip():
                        return value
                    elif isinstance(value, dict):
                        # Recurse into nested dict
                        nested = self._extract_text_from_json(value)
                        if nested:
                            return nested

            # Check for error
            if "error" in data:
                error_msg = data["error"]
                logger.warning(f"Cursor API error (full): {data}")
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", str(error_msg))
                logger.warning(f"Cursor API error: {error_msg}")

        return None

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
