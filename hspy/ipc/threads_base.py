# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:12:35 2024

@author: rehmer
"""

from threading import Thread, Event
from threading import Condition
import queue
from queue import Queue
import time
import types

from hspy.ipc.misc import (_new_write_profile, _print_profile_report,
                           drop_oldest_put, _GET_TIMEOUT_S)

class RWThread_R1(Thread):
    """
    Base class for a thread that reads from one buffer and writes into another.

    Subclasses read their input inside _target() and must do so through
    self._get(), never through self.read_buffer.get() -- see _get().
    """
    def __init__(self,
                 name : str,
                 read_buffer:Queue,
                 write_buffer:Queue,
                 get_timeout: float = _GET_TIMEOUT_S,
                 **kwargs):

        super().__init__(target = self._target,
                         **kwargs)

        self.name = name
        self.read_buffer = read_buffer
        self.write_buffer = write_buffer

        # Upper bound on how long one _get() waits for an item, i.e. on
        # how long stop() takes to be noticed while no data arrives.
        self.get_timeout = get_timeout

        self._exit = Event()

        self.daemon = True  # dies automatically if main thread exits




    def run(self):

        # Check if thread has been stopped
        while not self._exit.is_set():

            try:

                # Process data (the subclass reads read_buffer itself,
                # through self._get())
                result = self._target()

                # Put result into downstream buffer
                self.write_buffer.put(result)

            except queue.Empty:
                # Nothing arrived within get_timeout. Let the subclass do
                # its idle work, then re-check the exit flag at the top of
                # the loop instead of waiting here for an item that may
                # never come.
                self._on_idle()
                continue

            except Exception as e:
                print(f"[{self.name}] {e}")

    def _get(self, timeout: float = None):
        """
        Read one item from read_buffer, waiting at most `timeout` seconds
        (self.get_timeout by default) instead of forever.

        Subclasses must use this in _target() rather than calling
        self.read_buffer.get() directly: a bare get() blocks until an item
        arrives, which never happens once the stage upstream has ended, so
        the loop above would never see stop() and join() would hang. The
        queue.Empty raised on timeout is caught there and turned into
        another pass around the loop.
        """
        return self.read_buffer.get(timeout=self.get_timeout
                                    if timeout is None else timeout)

    def _on_idle(self):
        """Override in subclass. Called whenever a _get() timed out, i.e.
        while no data arrives -- for keeping a GUI responsive, handling
        commands, ..."""
        pass

    def _target(self):
        """Override in subclass. Should read from self.read_buffer via
        self._get()."""
        raise NotImplementedError

    def stop(self):
        self._exit.set()
                

class RWThread(Thread):
    """
    Base class for a thread that reads from one buffer and writes into another
    """
    def __init__(self,target,
                 read_buffer:Queue,
                 read_condition:Condition,
                 write_buffer:Queue,
                 write_condition:Condition,
                 **kwargs):

        self.read_buffer = read_buffer
        self.read_condition = read_condition
        
        self.write_buffer = write_buffer
        self.write_condition = write_condition
        
        self._exit = False               # Exit flag 

        super().__init__(target=target,**kwargs)
    
    def run(self):
        
        # Check if thread has been stopped
        while self._exit == False:
            
            # Acquire the read condition
            with self.read_condition:
                
                # Wait until the upstream thread notifies this thread
                while self.read_buffer.empty():
                    self.read_condition.wait()  
                
                # Execute target function (get item from read_buffer and process it)
                result = self._target()
                
                # Signal that processing on this item in the read_buffer is done
                # self.read_buffer.task_done()
                
                # Notify the upstream thread, that the item has been retrieved
                # from the buffer
                self.read_condition.notify()
            
            # Acquire the write condition
            with self.write_condition:
                
                # Write result to buffer
                self.write_buffer.put(result)
                
                # Notify the downstream thread, that item has been placed in the
                # write buffer
                self.write_condition.notify()
                
                # Wait for the downstream thread to process
                while not self.write_buffer.empty():
                    self.write_condition.wait()
            

            
    def _target_function(self):
        
        print('You need to overwrite this method in the child class.')
        # Get result from upstream thread
        upstream_dict = self.read_buffer.get()
    
        return upstream_dict
                        
    def stop(self):
        self._exit = True
        
  

class WThread(Thread):
    """
    Base class for a thread that writes into a buffer
    """
    def __init__(self,
                 target,
                 write_buffer:Queue,
                 write_condition:Condition,
                 **kwargs):
        
        self.write_buffer = write_buffer
        self.write_condition = write_condition
        self._exit = False 
        
        super().__init__(target=target,**kwargs)
        
    def run(self):
        
        # Check if thread has been stopped
        while self._exit == False:
                        
            # Execute target function
            result = self._target()
            
            # Acquire the write condition
            with self.write_condition:
            
                # Write result to buffer
                self.write_buffer.put(result)
                
                # Notify the downstream thread, that item has been placed in the
                # write buffer
                self.write_condition.notify()
                
                # Wait for the downstream thread to process
                while not self.write_buffer.empty():
                    self.write_condition.wait(timeout=0.5)
                       
    def _target_function(self):
        # This is a basic target function that produces a fake processing
        # result
        
        # Get result from upstream thread
        result_dict = {'fake':1}
        
        return result_dict
    
    
    def stop(self):
        self._exit = True
        
class WThread_R1(Thread):
    """
    Simplification of WThread. Solves two problems:
        1. Condition is redundant with Queue: queue.Queue is already
        thread-safe and has its own internal locking
        2. The wait-loop logic defeats the purpose of threading:
            self.write_condition.wait(timeout=0.5) makes WThread block until
            the downstream thread has consumed the item

    Mirrors WProcess_R1 (processes_base.py) feature for feature, so that a
    client class reads the same whether it runs in a thread or in a
    process: the same _setup() / _target() / _teardown() hooks, the same
    opt-in profiling, and the same drop-oldest backpressure policy.

    _setup() runs once at the top of run(), i.e. already inside this
    thread. Unlike a process, a thread shares the parent's memory, so it
    is technically free to build its resources in __init__ -- but doing
    the device work in _setup() means the caller's thread is not the one
    left holding a half-opened device if it fails, and it keeps the
    threading and multiprocessing versions of a client interchangeable.
    """

    def __init__(self,
                 name : str,
                 write_buffer: Queue,
                 profile: bool = False,
                 profile_every: int = 50,
                 **kwargs):

        super().__init__(target = self._target,
                         **kwargs)
        self.name = name
        self.write_buffer = write_buffer

        # profile=True prints a rolling-average timing report every
        # profile_every iterations: how long _target() took vs. how long
        # handing the result to write_buffer took, vs. the overall cycle
        # time. Off by default -- zero overhead, zero behaviour change
        # unless enabled.
        self.profile = profile
        self.profile_every = profile_every

        self._exit = Event()
        self.daemon = True  # dies automatically if main thread exits

    def _setup(self):
        """
        Override in subclass. Called exactly once, inside this thread,
        before the loop in run() starts. Start a stream, bind a device,
        reset counters, ... here rather than in __init__.
        """
        pass

    def run(self):

        # A _setup() that raises must not skip _teardown(): whatever it
        # managed to open before it failed still has to be released.
        try:
            self._setup()
        except Exception as e:
            print(f"[{self.name}] Setup failed: {e}")
            self._exit.set()

        if self.profile:
            prof = _new_write_profile()
            prof_i = 0
            prof_last_t = time.perf_counter()

        # Check if thread has been stopped
        while not self._exit.is_set():

            try:
                if self.profile:
                    t_cycle_start = time.perf_counter()

                    t0 = time.perf_counter()
                    result = self._target()
                    t1 = time.perf_counter()

                    self._put(result)
                    t2 = time.perf_counter()

                    prof['target'].add(t1 - t0)
                    prof['put'].add(t2 - t1)
                    prof['cycle'].add(t_cycle_start - prof_last_t)
                    prof_last_t = t_cycle_start

                    prof_i += 1
                    if prof_i % self.profile_every == 0:
                        _print_profile_report(self.name, prof)

                else:
                    # Execute target function
                    result = self._target()

                    # Write result to buffer (never blocks: drops the
                    # oldest item if the consumer cannot keep up)
                    self._put(result)

            except queue.Empty:
                continue

            except Exception as e:
                print(f"[{self.name}] {e}")

        self._teardown()

    def _put(self, result) -> bool:
        """
        Hand one result to write_buffer using the drop-oldest policy --
        see misc.drop_oldest_put() for why a producer must never block
        here. Returns True if the item was placed.
        """
        return drop_oldest_put(self.write_buffer, result, self.name)

    def _teardown(self):
        """Override in subclass for optional cleanup inside this thread
        once the loop has exited (e.g. stopping a stream)."""
        pass

    def _target(self):
        """Override in subclass to produce data."""
        raise NotImplementedError

    def stop(self):
        self._exit.set()
        
        
class RThread(Thread):
    """
    Base class for a thread that reads from a buffer
    """
    def __init__(self,
                 target,
                 read_buffer:Queue,
                 read_condition:Condition,
                 **kwargs):
        
        self.read_buffer = read_buffer
        self.read_condition = read_condition
        self._exit = False 
        
        super().__init__(target=target,
                         **kwargs)
        
    def run(self):

        # Check if thread has been stopped
        while self._exit == False:

            # Acquire the read condition
            with self.read_condition:

                # Wait until the upstream thread notifies this thread.
                # stop() sets _exit and notifies, so the flag has to be
                # re-checked here: otherwise a stopped thread goes back to
                # waiting for an item that will never arrive (this class
                # is not a daemon, so that hangs the interpreter at exit).
                while self.read_buffer.empty() and not self._exit:
                    self.read_condition.wait(timeout=0.5)

                if self._exit:
                    break

                # Execute target function
                result = self._target()
               
                # Notify the upstream thread, that the item has been retrieved
                # from the buffer
                self.read_condition.notify()
                
                       
    def _target_function(self):

        # Get result from upstream thread
        upstream_dict = self.read_buffer.get()
    
        return upstream_dict
            
    def stop(self):
        with self.read_condition:
            self._exit = True
            self.read_condition.notify_all()
            
            
class RThread_R1(Thread):
    """
    Base class for a thread that reads from a buffer (a sink stage with no
    downstream buffer). Mirrors RProcess_R1.

    Subclasses read their input inside _target() and must do so through
    self._get(), never through self.read_buffer.get() -- see _get().
    """
    def __init__(self,
                 name : str,
                 read_buffer:Queue,
                 get_timeout: float = _GET_TIMEOUT_S,
                 **kwargs):
        
        # Check input arguments
        if not isinstance(name,str):
            raise TypeError(f'name is type {type(name)} instead of {str}.')
        if not isinstance(read_buffer,Queue):
            raise TypeError(f'read_buffer is type {type(read_buffer)} instead of {Queue}.')
        if not isinstance(name,str):
            raise TypeError(f'self._target is type {type(self._target)} instead of {types.FunctionType}.')
        
        # Call __init__ of parent class
        super().__init__(**kwargs,
                         target = self._target)
        
        # Assign user specified attributes
        self.name = name
        self.read_buffer = read_buffer

        # Upper bound on how long one _get() waits for an item, i.e. on
        # how long stop() takes to be noticed while no data arrives.
        self.get_timeout = get_timeout

        # Standard attributes
        self._exit = Event()
        self.daemon = True  # dies automatically if main thread exits



    def run(self):

        # Check if thread has been stopped
        while not self._exit.is_set():

            try:
                # Execute target function (reads read_buffer itself,
                # through self._get())
                result = self._target()

            except queue.Empty:
                # Nothing arrived within get_timeout. Let the subclass do
                # its idle work, then re-check the exit flag at the top of
                # the loop instead of waiting here for an item that may
                # never come.
                self._on_idle()
                continue

            except Exception as e:
                print(f"[{self.name}] {e}")

    def _get(self, timeout: float = None):
        """
        Read one item from read_buffer, waiting at most `timeout` seconds
        (self.get_timeout by default) instead of forever.

        Subclasses must use this in _target() rather than calling
        self.read_buffer.get() directly: a bare get() blocks until an item
        arrives, which never happens once the stage upstream has ended, so
        the loop above would never see stop() and join() would hang. The
        queue.Empty raised on timeout is caught there and turned into
        another pass around the loop.
        """
        return self.read_buffer.get(timeout=self.get_timeout
                                    if timeout is None else timeout)

    def _on_idle(self):
        """Override in subclass. Called whenever a _get() timed out, i.e.
        while no data arrives -- for keeping a GUI responsive, handling
        commands, ..."""
        pass

    def _target(self):
        """Override in subclass. Should read from self.read_buffer via
        self._get()."""

        raise NotImplementedError

    def stop(self):
        self._exit.set()

            
