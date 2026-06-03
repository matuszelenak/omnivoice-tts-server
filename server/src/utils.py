async def async_enumerate(aiterable, start=0):
    index = start
    async for value in aiterable:
        yield index, value
        index += 1
