
import sys
sys.path.append("../../")
from lib import send_funcs as sf
#import npp_funcs as nppf
import yaml
def int_eval(data):
    return eval(str(data))
if __name__ == "__main__":
    f = open('./general_config.yaml','r')
    params = yaml.safe_load(f)
    f.close()
    decimation = int_eval(params['decimation'])

    pg = sf.get_pg(params)

    no_subplot_measurements = ["echo", "echo_1ax"]
    subplots = not params['measurement'] in no_subplot_measurements

    #subplots = True
    #if params['measurement'] == "echo" or params['measurement'] == "echo_1ax":
    #    subplots = False

    pg.show(decimation, subplots)
