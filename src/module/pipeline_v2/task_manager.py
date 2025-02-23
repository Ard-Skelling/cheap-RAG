import asyncio


class CoroTaskManager:
    def __init__(self):
        self.tasks = dict()    # 存储任务字典
        self.running = True    # 控制任务监控是否继续运行
        self.events = dict()    # 事件同步
        self.results = dict()


    async def add_task(self, task_id, task_coroutine):
        """添加新任务，接受任务ID和任务协程"""
        event = asyncio.Event()
        self.events[task_id] = event

        async def wrapped_task():
            try:
                result = await task_coroutine
                self.results[task_id] = result
            except Exception as e:
                self.results[task_id] = e
            finally:
                event.set()

        task = asyncio.create_task(wrapped_task())
        self.tasks[task_id] = task


    async def wait_result(self, task_id, timeout):
        """等待任务执行结果"""
        event: asyncio.Event = self.events.pop(task_id, None)
        if event:
            try:
                await asyncio.wait_for(event.wait(), timeout)
                result = self.results.pop(task_id)
                if isinstance(result, Exception):
                    raise result
                return result
            except asyncio.TimeoutError:
                raise TimeoutError(f'Task {task_id} timed out after {timeout} seconds.')
        raise ValueError(f"Task {task_id} doesn't exist.")


    async def remove_completed_tasks(self):
        """移除已完成的任务
        TODO: 未来使用数据库来存储任务结果"""
        done, pending = await asyncio.wait(self.tasks.values(), return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            # 获取任务 ID 和结果
            task_id, task_result = task.result()
            print(f'Task {task_id} is done!')

            # 从字典中移除任务
            for task_id, t in list(self.tasks.items()):
                if t == task:
                    task = self.tasks.pop(task_id)
                    break


    async def monitor_tasks(self):
        """持续监控并清理已完成的任务，直到程序结束。"""
        while self.running:
            await asyncio.sleep(1)    # 每秒检查一次任务状态
            if self.tasks:
                await self.remove_completed_tasks()


    def start(self):
        """启动任务监控的协程
        用一个线程来运行此协程，可以实现任务的后台调度。示例如下：
        import asyncio
        from concurrent.futures import ProcessPoolExecutor

        async def add_new_task(task_manager, task_func, *args, **kwargs):
            ...

        async def main():
            loop = asyncio.get_running_loop()
            with ProcessPoolExecutor(4) as pool:
                task_manager = CoroTaskManager()
                
                # 启动任务监控
                monitor_task = task_manager.start()

                # 动态添加任务
                await add_new_task(task_manager, task_func, *args, **kwargs)
                await add_new_task(task_manager, task_func, *args, **kwargs)

                # 延迟一段时间后继续添加任务
                await asyncio.sleep(2)
                await add_new_task(task_manager, task_func, *args, **kwargs)
                
                # 等待一些时间来观察任务执行
                await asyncio.sleep(10)

                # 可以手动停止任务监控，不然任务管理器会一直运行
                # task_manager.stop()

                # 等待所有任务完成
                await monitor_task

        """
        return asyncio.create_task(self.monitor_tasks())


    def stop(self):
        """停止任务监控"""
        self.running = False