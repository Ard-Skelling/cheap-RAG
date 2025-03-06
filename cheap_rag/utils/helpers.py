import base64
import time
import threading
import hashlib
import asyncio
import importlib
from functools import wraps


# Local modules
from utils.logger import logger


def lazy_import(module_name):
    """Lazy import lib into certain function. Example:
    @lazy_import(torch)
    def func_use_torch(a, b, torch=None):
        result = torch.tensor([1, 2])
        return result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            module = importlib.import_module(module_name)
            logger.debug(f'Dynamically imported: {module_name}')
            kwargs[module_name] = module
            return func(*args, **kwargs)
        return wrapper
    return decorator


def ftimer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        st = time.time()
        result = func(*args, **kwargs)
        et = time.time()
        logger.debug(f'function {func.__name__} cost time: {et - st}')
        return result
    return wrapper


def atimer(coro):
    @wraps(coro)
    async def wrapper(*args, **kwargs):
        st = time.time()
        result = await coro(*args, **kwargs)
        et = time.time()
        logger.debug(f'coroutine {coro.__name__} cost time: {et - st}')
        return result
    return wrapper


def b64_to_bytes(b64_data:str, encoding="utf-8"):
    return base64.b64decode(b64_data.encode(encoding))


def bytes_to_b64(bytes_data:bytes, encoding="utf-8"):
    return base64.b64encode(bytes_data).decode(encoding)


generate_md5 = lambda text: hashlib.md5(text.encode('utf-8')).hexdigest()


# metaclass singleton used for singleton instance
class Singleton(type):
    """Singleton metaclass. Make sure there is only one class instance exists. Example:

    class MySingleton(metaclass=Singleton):
        __allow_reinitialization = False    # Allow reinitiatialization or not

        def __init__(self, value):
            print("Initializing MySingleton")
            self.value = value

        def display(self):
            print(f"Singleton value: {self.value}")
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # we have not every built an instance before.  Build one now.
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        else:
            instance = cls._instances[cls]
            # here we are going to call the __init__ and maybe reinitialize.
            if hasattr(cls, '__allow_reinitialization') and cls.__allow_reinitialization:
                # if the class allows reinitialization, then do it
                instance.__init__(*args, **kwargs)  # call the init again
        return instance
    

class SnowflakeIDGenerator:
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.sequence = 0
        self.last_timestamp = -1
        self.lock = threading.Lock()

        # Bits allocation
        self.timestamp_bits = 41
        self.machine_id_bits = 10
        self.sequence_bits = 12

        # Max values
        self.max_machine_id = (1 << self.machine_id_bits) - 1
        self.max_sequence = (1 << self.sequence_bits) - 1

        # Shifts
        self.timestamp_shift = self.machine_id_bits + self.sequence_bits
        self.machine_id_shift = self.sequence_bits

        if self.machine_id > self.max_machine_id or self.machine_id < 0:
            raise ValueError(f"Machine ID must be between 0 and {self.max_machine_id}")

    def _current_timestamp(self):
        return int(time.time() * 1000)

    def _wait_for_next_millisecond(self, last_timestamp):
        timestamp = self._current_timestamp()
        while timestamp <= last_timestamp:
            timestamp = self._current_timestamp()
        return timestamp

    def generate_id(self):
        with self.lock:
            timestamp = self._current_timestamp()

            if timestamp < self.last_timestamp:
                raise Exception("Clock moved backwards. Refusing to generate ID.")

            if timestamp == self.last_timestamp:
                self.sequence = (self.sequence + 1) & self.max_sequence
                if self.sequence == 0:
                    timestamp = self._wait_for_next_millisecond(self.last_timestamp)
            else:
                self.sequence = 0

            self.last_timestamp = timestamp

            id = ((timestamp << self.timestamp_shift) |
                  (self.machine_id << self.machine_id_shift) |
                  self.sequence)
            return id


class AsyncDict:
    def __init__(self, max_size):
        self.max_size = max_size
        self._dict = dict()
        self.lock = asyncio.Lock()  # 用于线程安全
        self.not_full = asyncio.Condition(self.lock)  # 用于等待字典未满

    async def put(self, key, value):
        async with self.lock:
            # 如果字典已满，等待直到有空间
            while len(self._dict) >= self.max_size:
                logger.debug(f"Dictionary is full. Waiting to insert item {key}\n{value}")
                await self.not_full.wait()  # 阻塞等待

            # 插入新键值对
            self._dict[key] = value
            logger.debug(f"Inserted item {key}\n{value}")

    async def remove(self, key):
        async with self.lock:
            if key in self._dict:
                # 移除键值对
                value = self._dict.pop(key)
                logger.debug(f"Removed key: {key}\n{value}")
                # 通知等待的协程字典未满
                self.not_full.notify_all()
            else:
                logger.debug(f"Key {key} not found in dictionary.")

    async def get(self, key):
        async with self.lock:
            return self._dict.get(key)


    async def pop(self, key, default=None):
        async with self.lock:
            result = self._dict.pop(key, default)
            self.not_full.notify_all()
        return result

    def __contains__(self, key):
        return key in self._dict

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return repr(self._dict)
    

# Example usage
if __name__ == "__main__":
    generator = SnowflakeIDGenerator(machine_id=1)
    for _ in range(10):
        logger.info(generator.generate_id())