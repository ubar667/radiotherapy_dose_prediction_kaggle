import zipfile
import os
from tqdm import tqdm
# Function : file_compress
#https://www.tutorialspoint.com/how-to-compress-files-with-zipfile-module-in-python#:~:text=To%20create%20your%20own%20compressed,it%20into%20the%20ZIP%20file.
def file_compress(inp_file_names, out_zip_file):
    """
    function : file_compress
    args : inp_file_names : list of filenames to be zipped
    out_zip_file : output zip file
    return : none
    assumption : Input file paths and this code is in same directory.
    """
    # Select the compression mode ZIP_DEFLATED for compression
    # or zipfile.ZIP_STORED to just store the file
    compression = zipfile.ZIP_DEFLATED

    # create the zip file first parameter path/name, second mode
    out_zip_file = out_zip_file + ".zip"
    print(f' *** out_zip_file is - {out_zip_file}')
    zf = zipfile.ZipFile(out_zip_file, mode="w")

    try:
        for file_to_write in tqdm(inp_file_names):
            # Add file to the zip file
            # first parameter file to zip, second filename in zip
            file_name = os.path.basename(file_to_write)
            zf.write(file_to_write, file_name, compress_type=compression)

    except FileNotFoundError as e:
        print(f' *** Exception occurred during zip process - {e}')
    finally:
        # Don't forget to close the file!
        zf.close()