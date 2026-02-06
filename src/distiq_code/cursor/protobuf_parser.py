"""Protobuf parser and modifier for Cursor API requests.

Parses StreamUnifiedChatWithToolsRequest to extract model and message content,
then modifies model field for smart routing.
"""

import gzip
import struct
from typing import Optional

from loguru import logger


class ProtobufParser:
    """
    Parse and modify Cursor API protobuf requests.

    Handles ConnectRPC envelope format and protobuf field extraction.
    """

    @staticmethod
    def decode_varint(data: bytes, offset: int = 0) -> tuple[int, int]:
        """
        Decode protobuf varint.

        Args:
            data: Binary data
            offset: Starting offset

        Returns:
            Tuple of (value, new_offset)
        """
        value = 0
        shift = 0
        pos = offset

        while pos < len(data):
            byte = data[pos]
            value |= (byte & 0x7F) << shift
            pos += 1

            if not (byte & 0x80):
                break

            shift += 7

        return value, pos

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
    def parse_envelope(body: bytes) -> tuple[bool, bytes]:
        """
        Parse ConnectRPC envelope.

        Format: [flags:1][length:4_BE][protobuf_payload]

        Args:
            body: Raw request body

        Returns:
            Tuple of (is_compressed, protobuf_payload)
        """
        if len(body) < 5:
            raise ValueError("Body too short for envelope")

        flags = body[0]
        length = struct.unpack(">I", body[1:5])[0]
        protobuf_payload = body[5:]

        is_compressed = flags == 0x01

        # Decompress if needed
        if is_compressed:
            protobuf_payload = gzip.decompress(protobuf_payload)

        return is_compressed, protobuf_payload

    @staticmethod
    def wrap_envelope(protobuf_payload: bytes, compress: bool = False) -> bytes:
        """
        Wrap protobuf in ConnectRPC envelope.

        Args:
            protobuf_payload: Protobuf binary data
            compress: Apply gzip compression

        Returns:
            Enveloped body
        """
        if compress:
            protobuf_payload = gzip.compress(protobuf_payload)
            flags = 0x01
        else:
            flags = 0x00

        length = len(protobuf_payload)
        envelope = bytes([flags]) + struct.pack(">I", length) + protobuf_payload

        return envelope

    @staticmethod
    def extract_model(protobuf_data: bytes) -> Optional[str]:
        """
        Extract model name from StreamUnifiedChatWithToolsRequest.

        Walks through protobuf fields to find Model.name (field 5 → field 1).

        Args:
            protobuf_data: Protobuf payload

        Returns:
            Model name or None
        """
        try:
            offset = 0

            while offset < len(protobuf_data):
                # Read field tag
                tag, offset = ProtobufParser.decode_varint(protobuf_data, offset)
                field_num = tag >> 3
                wire_type = tag & 0x07

                # Field 1 = Request (nested message)
                if field_num == 1 and wire_type == 2:
                    # Read length
                    length, offset = ProtobufParser.decode_varint(protobuf_data, offset)
                    request_data = protobuf_data[offset : offset + length]

                    # Parse Request to find Model (field 5)
                    model_name = ProtobufParser._extract_model_from_request(request_data)
                    if model_name:
                        return model_name

                    offset += length

                # Skip other fields
                elif wire_type == 0:  # Varint
                    _, offset = ProtobufParser.decode_varint(protobuf_data, offset)
                elif wire_type == 1:  # 64-bit
                    offset += 8
                elif wire_type == 2:  # Length-delimited
                    length, offset = ProtobufParser.decode_varint(protobuf_data, offset)
                    offset += length
                elif wire_type == 5:  # 32-bit
                    offset += 4
                else:
                    logger.warning(f"Unknown wire type: {wire_type}")
                    break

        except Exception as e:
            logger.debug(f"Failed to extract model: {e}")

        return None

    @staticmethod
    def _extract_model_from_request(request_data: bytes) -> Optional[str]:
        """Extract model name from Request message (field 5)."""
        try:
            offset = 0

            while offset < len(request_data):
                tag, offset = ProtobufParser.decode_varint(request_data, offset)
                field_num = tag >> 3
                wire_type = tag & 0x07

                # Field 5 = Model (nested message)
                if field_num == 5 and wire_type == 2:
                    length, offset = ProtobufParser.decode_varint(request_data, offset)
                    model_data = request_data[offset : offset + length]

                    # Parse Model to find name (field 1)
                    return ProtobufParser._extract_string_field(model_data, field_num=1)

                # Skip other fields
                elif wire_type == 0:
                    _, offset = ProtobufParser.decode_varint(request_data, offset)
                elif wire_type == 1:
                    offset += 8
                elif wire_type == 2:
                    length, offset = ProtobufParser.decode_varint(request_data, offset)
                    offset += length
                elif wire_type == 5:
                    offset += 4

        except Exception as e:
            logger.debug(f"Failed to extract model from request: {e}")

        return None

    @staticmethod
    def _extract_string_field(data: bytes, field_num: int) -> Optional[str]:
        """Extract string value from protobuf field."""
        try:
            offset = 0

            while offset < len(data):
                tag, offset = ProtobufParser.decode_varint(data, offset)
                current_field = tag >> 3
                wire_type = tag & 0x07

                if current_field == field_num and wire_type == 2:
                    # String field (length-delimited)
                    length, offset = ProtobufParser.decode_varint(data, offset)
                    string_bytes = data[offset : offset + length]
                    return string_bytes.decode("utf-8", errors="ignore")

                # Skip field
                if wire_type == 0:
                    _, offset = ProtobufParser.decode_varint(data, offset)
                elif wire_type == 1:
                    offset += 8
                elif wire_type == 2:
                    length, offset = ProtobufParser.decode_varint(data, offset)
                    offset += length
                elif wire_type == 5:
                    offset += 4

        except Exception as e:
            logger.debug(f"Failed to extract string field: {e}")

        return None

    @staticmethod
    def replace_model(protobuf_data: bytes, new_model: str) -> bytes:
        """
        Replace model name in StreamUnifiedChatWithToolsRequest.

        Creates new protobuf with substituted model field.

        Args:
            protobuf_data: Original protobuf
            new_model: New model name

        Returns:
            Modified protobuf
        """
        try:
            # This is complex - need to rebuild entire protobuf structure
            # For now, use a simpler approach: find and replace model string

            # Extract old model
            old_model = ProtobufParser.extract_model(protobuf_data)
            if not old_model:
                logger.warning("Could not extract old model, skipping replacement")
                return protobuf_data

            # Simple string replacement (works if model appears once)
            old_model_bytes = old_model.encode("utf-8")
            new_model_bytes = new_model.encode("utf-8")

            # Find model field pattern: tag (field 1, wire type 2) + length + string
            # Tag for Model.name (field 5 → field 1) = 0x0a (field 1, wire 2)
            old_pattern = bytes([0x0A, len(old_model_bytes)]) + old_model_bytes
            new_pattern = bytes([0x0A, len(new_model_bytes)]) + new_model_bytes

            # Replace
            modified = protobuf_data.replace(old_pattern, new_pattern)

            if modified == protobuf_data:
                logger.warning("Model replacement pattern not found, returning original")

            return modified

        except Exception as e:
            logger.error(f"Failed to replace model: {e}")
            return protobuf_data

    @staticmethod
    def extract_messages(protobuf_data: bytes) -> list[str]:
        """
        Extract user message content from request.

        Args:
            protobuf_data: Protobuf payload

        Returns:
            List of message contents
        """
        messages = []

        try:
            offset = 0

            while offset < len(protobuf_data):
                tag, offset = ProtobufParser.decode_varint(protobuf_data, offset)
                field_num = tag >> 3
                wire_type = tag & 0x07

                # Field 1 = Request
                if field_num == 1 and wire_type == 2:
                    length, offset = ProtobufParser.decode_varint(protobuf_data, offset)
                    request_data = protobuf_data[offset : offset + length]
                    messages.extend(ProtobufParser._extract_messages_from_request(request_data))
                    offset += length

                # Skip other fields
                elif wire_type == 0:
                    _, offset = ProtobufParser.decode_varint(protobuf_data, offset)
                elif wire_type == 1:
                    offset += 8
                elif wire_type == 2:
                    length, offset = ProtobufParser.decode_varint(protobuf_data, offset)
                    offset += length
                elif wire_type == 5:
                    offset += 4

        except Exception as e:
            logger.debug(f"Failed to extract messages: {e}")

        return messages

    @staticmethod
    def _extract_messages_from_request(request_data: bytes) -> list[str]:
        """Extract message contents from Request.messages (field 1, repeated)."""
        messages = []

        try:
            offset = 0

            while offset < len(request_data):
                tag, offset = ProtobufParser.decode_varint(request_data, offset)
                field_num = tag >> 3
                wire_type = tag & 0x07

                # Field 1 = messages (repeated)
                if field_num == 1 and wire_type == 2:
                    length, offset = ProtobufParser.decode_varint(request_data, offset)
                    message_data = request_data[offset : offset + length]

                    # Extract content from Message (field 1)
                    content = ProtobufParser._extract_string_field(message_data, field_num=1)
                    if content:
                        messages.append(content)

                    offset += length

                # Skip other fields
                elif wire_type == 0:
                    _, offset = ProtobufParser.decode_varint(request_data, offset)
                elif wire_type == 1:
                    offset += 8
                elif wire_type == 2:
                    length, offset = ProtobufParser.decode_varint(request_data, offset)
                    offset += length
                elif wire_type == 5:
                    offset += 4

        except Exception as e:
            logger.debug(f"Failed to extract messages from request: {e}")

        return messages
