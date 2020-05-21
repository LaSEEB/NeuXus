from pylsl import StreamInlet, resolve_stream
import asyncio

async def resolve_streams(type_, arg_):
    a = resolve_stream(type_, arg_)
    return

async def get_marker_stream():
    # other ways to query a stream, too - for instance by content-type)
    await resolve_streams("name", "MarkerStream")
    results = resolve_stream("name", "MarkerStream")

    # open an inlet so we can read the stream's data (and meta-data)
    inlet = StreamInlet(results[0])

    # get the full stream info (including custom meta-data) and dissect it
    info = inlet.info()
    print("The stream's XML meta-data is: ")
    print(info.as_xml())

async def get_openvibeSignal():
    await resolve_streams("name", "openvibeSignal")
    results = resolve_stream("name", "openvibeSignal")

    # open an inlet so we can read the stream's data (and meta-data)
    inlet = StreamInlet(results[0])

    # get the full stream info (including custom meta-data) and dissect it
    info = inlet.info()
    print("The stream's XML meta-data is: ")
    print(info.as_xml())

async def main():
    await asyncio.gather(get_marker_stream(), get_openvibeSignal())
    print('I get both signal info')

if __name__ == "__main__":
    import time
    s = time.perf_counter()
    asyncio.run(main())
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")

