from io import BytesIO
from zipfile import ZipFile
class ZipFileManager:
    def __init__(self, zip_bytes):
        self.zip_bytes = zip_bytes

    def list_contents(self):
        """
        List all files and directories in the ZIP archive.
        """
        with ZipFile(self.zip_bytes, 'r') as zip_ref:
            return zip_ref.namelist()

    def open_file(self, file_path):
        """
        Read the content of a specific file within the ZIP archive.
        """
        with ZipFile(self.zip_bytes, 'r') as zip_ref:
            try:
                file = zip_ref.open(file_path)
                return file
            except KeyError:
                raise FileNotFoundError(f"File '{file_path}' not found in ZIP archive.")

    def open_raster(self, file_path):
        """
        Read the content of a specific file within the ZIP archive.
        """
        with ZipFile(self.zip_bytes, 'r') as zip_ref:
            try:
                file = BytesIO(zip_ref.read(file_path))
                return file
            except KeyError:
                raise FileNotFoundError(f"File '{file_path}' not found in ZIP archive.")

    def get_bytes(self):
        return self.zip_bytes