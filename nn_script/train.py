from TensorflowToolbox.utility import read_proto as rp
from net_flow import NetFlow

#from model import IIGan as model

if __name__ == "__main__":
	model_params = rp.load_proto("model.tfproto")
	net = NetFlow(model_params, True, True)
	net.mainloop()
