# -*- coding: utf-8 -*-
"""
Created on Fri Mar 3 10:51:15 2023

@author: lqc
"""

class Instrument:
    
    def __init__(self, resource_man, addr):
        """
        Initialize the Instrument object.

        Args:
            resource_man: The VISA resource manager which is visa.ResourceManager().
            addr (str): The address of the instrument. 
        """
        self.rm = resource_man
        self.inst = self.rm.open_resource(addr)
        self.inst.timeout = None
        
    def write(self, msg):
        """
        Write a command to the instrument.

        Args:
            msg (str): The command to be sent to the instrument.
        """
        return self.inst.write(msg)
        
    def query(self, msg):
        """
        Send a query command to the instrument and receive the response.

        Args:
            msg (str): The query command to be sent to the instrument.

        Returns:
            str: The response received from the instrument.
        """
        return self.inst.write(msg)

    def close(self):
        """
        Close the connection to the instrument.
        """
        self.inst.close()
