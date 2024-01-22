import numpy as np



def tapping(
    num_ifos,
    mask
):
    
    if num_ifos != 2:
        
        raise ValueError(f"Number of ifos needs to be 2, the assigned value is {num_ifos}.")
    
    tape = np.zeros([num_ifos, mask.size])
    # To Do expand this to mutipule ifo use case.
    for i in np.unique(mask):
        idx = np.where(mask == i)[0]
        if i == 0:
            tape[:,idx] = np.array([[0], [0]])
            
        if i == 1:
            tape[:,idx] = np.array([[0], [1]])

        if i == 2:
            tape[:,idx] = np.array([[1], [0]])
            
        if i == 3:
            tape[:,idx] = np.array([[1], [1]])
            
    return tape