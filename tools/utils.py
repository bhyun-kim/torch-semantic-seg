
def cvt_pathToModule(file_path):
    """Convert path (string) to module form.
    
    Args :
        file_path (str) : file path written in nomal path form
    
    Returns :
        module_form (str) : file path in module form (i.e. matplotlib.pyplot)
    
    """
    
    file_path = file_path.replace('/', '.')
    module_form  = file_path.replace('.py', '')
    
    return module_form

def cvt_moduleToDict(mod) :
    """
    Args : 
        mod (module)  
    
    Returns :
        cfg (dict)
    
    """
    cfg = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
        }
    
    return cfg

