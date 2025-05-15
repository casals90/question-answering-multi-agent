import base64
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
