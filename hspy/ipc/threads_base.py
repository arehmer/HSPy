# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:12:35 2024

@author: rehmer
"""

from threading import Thread
from threading import Condition
from queue import Queue



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
                    self.write_condition.wait()
                       
    def _target_function(self):
        # This is a basic target function that produces a fake processing
        # result
        
        # Get result from upstream thread
        result_dict = {'fake':1}
        
        return result_dict
    
    
    def stop(self):
        self._exit = True
        
        
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
        
        super().__init__(target=target,**kwargs)
        
    def run(self):
        
        # Check if thread has been stopped
        while self._exit == False:
                        
            # Acquire the read condition
            with self.read_condition:
        
                # Wait until the upstream thread notifies this thread
                while self.read_buffer.empty():
                    self.read_condition.wait()     
        
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
        self._exit = True