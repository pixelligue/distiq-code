"""Generate self-signed SSL certificate for Cursor MITM proxy."""

import datetime
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from loguru import logger


def generate_ssl_cert(
    output_dir: Path = Path("."),
    days_valid: int = 365,
) -> tuple[Path, Path]:
    """
    Generate self-signed SSL certificate for Cursor API MITM.

    Args:
        output_dir: Directory to save certificate files
        days_valid: Certificate validity period in days

    Returns:
        Tuple of (cert_path, key_path)
    """
    logger.info("Generating self-signed SSL certificate...")

    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Certificate subject and issuer (same for self-signed)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "api2.cursor.sh"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Distiq Code Proxy"),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Development"),
    ])

    # Build certificate
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=days_valid))
        # Subject Alternative Names (required for modern browsers/clients)
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("api2.cursor.sh"),
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )

    # Save certificate
    cert_path = output_dir / "cursor_cert.pem"
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    # Save private key
    key_path = output_dir / "cursor_key.pem"
    with open(key_path, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    logger.success(f"SSL certificate generated: {cert_path}, {key_path}")
    logger.warning(
        "This is a self-signed certificate. "
        "Cursor may show SSL warnings - you'll need to accept them."
    )

    return cert_path, key_path


if __name__ == "__main__":
    generate_ssl_cert()
