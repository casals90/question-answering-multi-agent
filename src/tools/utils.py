import base64
import random
import string
import time
import uuid
from pathlib import Path


def image_to_base64(
        image_path: str | Path, include_mime_prefix: bool = True,
        return_mime_type: bool = False) -> str | tuple[str, str]:
    """
    Convert an image file to base64 encoded string.

    Parameters:
    -----------
    image_path : str or Path
        Path to the image file
    include_mime_prefix : bool, default=True
        If True, includes the data URI scheme
        prefix (e.g., "data:image/png;base64,")
    return_mime_type : bool, default=False
        If True, also returns the detected MIME type as a second return value

    Returns:
    --------
    str or Tuple[str, str]
        Base64 encoded string (with optional MIME prefix)
        If return_mime_type is True, returns a tuple of
        (base64_string, mime_type)

    Raises:
    -------
    FileNotFoundError
        If the image file doesn't exist
    ValueError
        If the file extension is not supported
    """
    # Convert to Path object for easier manipulation
    path = Path(image_path)

    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Get file extension and determine MIME type
    extension = path.suffix.lower()
    mime_mapping = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.svg': 'image/svg+xml',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
        '.ico': 'image/x-icon'
    }

    mime_type = mime_mapping.get(extension)
    if mime_type is None:
        raise ValueError(f"Unsupported image format: {extension}")

    # Read the binary data and encode as base64
    with open(path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    # Add MIME prefix if requested
    if include_mime_prefix:
        encoded_string = f"data:{mime_type};base64,{encoded_string}"

    # Return the result
    if return_mime_type:
        return encoded_string, mime_type
    else:
        return encoded_string


def generate_unique_id(
        prefix: str = "",
        length: int = 8,
        include_timestamp: bool = True
) -> str:
    """
    Generate a unique identifier.

    Args:
        prefix (str): Optional string to prefix the ID with
        length (int): Length of the random portion of the ID (default: 8)
        include_timestamp (bool): Whether to include timestamp
            in the ID (default: True)

    Returns:
        str: A unique ID string
    """

    # Start with the prefix if provided
    unique_id = prefix

    # Add timestamp if requested
    if include_timestamp:
        unique_id += str(int(time.time())) + "_"

    # Generate random string of specified length
    if length > 0:
        # Use uuid4 for the first part (which is already random)
        random_part = str(uuid.uuid4()).replace("-", "")

        # If we need more characters than uuid4 provides, add random
        # alphanumeric chars
        if length > 32:
            extra_chars = ''.join(random.choices(
                string.ascii_letters + string.digits,
                k=length - 32
            ))
            random_part += extra_chars

        # Trim if requested length is less than what uuid4 provides
        random_part = random_part[:length]

        unique_id += random_part

    return unique_id
