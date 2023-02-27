from NPR_structures import *

def ccextrap(data_dict, ensembles=None, **kwargs):
    if ensembles==None:
        ensembles = list(data_dict.keys())

    extrap_dict = {e:{'m_pi'
