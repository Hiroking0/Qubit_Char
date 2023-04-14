
class Instrument:

    #resource_man should be from visa.ResourceManager()
    def __init__(self, resource_man, addr):
        self.rm = resource_man
        self.inst = self.rm.open_resource(addr)
        self.inst.timeout = None


    def write(self, msg):
        return self.inst.write(msg)
        
    def query(self, msg):
        return self.inst.write(msg)

    def read(self):
        return self.inst.read()
    
    def close(self):
        self.inst.close()
    
    
    
