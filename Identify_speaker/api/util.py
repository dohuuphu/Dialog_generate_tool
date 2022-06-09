import aiofiles


async def save_file(_file,file_path):
        contents = await _file.read()
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(contents)
        return file_path