import os
import sys
import json
import shutil
import torch

###
# for gpu setup
###
def gpu_setup(gpu_id):
    #set up GPUS
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    if gpu_id == "auto":
        try:
            #try to find empty gpu automaticaly
            import GPUtil
            gpu_id = GPUtil.getFirstAvailable(order = 'memory', maxLoad=0.5, maxMemory=0.5, attempts=1, interval=900, verbose=False)
            gpu_id = gpu_id[0]
        except:
            print("can't import GPUtil. maybe you can do: pip install gputil")
            print("gpu id is set to -1")
            gpu_id = -1
    
    gpu_id = int(gpu_id)
    if gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("gpu id: %s"%gpu_id)
    print("using device: %s"%device)
    return device

###
# for logging
###
def savedir_setup(savedir,basedir="../experiments",args=None,append_args=[]):
    '''
    setup savedir (save directory) with time. 
    Input: base savedir name if savedir=no it will create temp dir
    Return: savedir name with time
    '''
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if savedir=="no":
        savedir = os.path.join("/tmp/",savedir+"_"+now)
    else:
        if len(append_args) > 0 and args is not None: 
            for arg_opt in append_args:
                arg_value = getattr(args, arg_opt)
                savedir+="_"+arg_opt+"-"+str(arg_value)

        savedir = os.path.join(basedir,savedir+"_"+now)
    
    #make savedir
    os.makedirs(savedir)
    print("made the log directory",savedir)

    return savedir

def save_args(savedir,args,name="args.json"):
    #save args as "args.json" in the savedir
    with open(os.path.join(savedir,name), 'w') as f:
        json.dump( vars(args), f, sort_keys=True, indent=4)
        
def save_json(dict,path):
    with open(path, 'w') as f:
        json.dump( dict, f, sort_keys=True, indent=4)
        print("log saved at %s"%path)

def check_githash():
    import warnings
    try:
        import git
    except:
        print("cannot import gitpython; try pip install gitpython")
        return None
    
    #from https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
    #from https://stackoverflow.com/questions/31540449/how-to-check-if-a-git-repo-has-uncommitted-changes-using-python
    try:
        repo = git.Repo(search_parent_directories=True)
        if repo.is_dirty():
            warnings.warn("WARNNING! the current git repository is dirty! Do not use for formal experiments")
        sha = repo.head.object.hexsha
        return sha
    except:
        print("cannot get githash")
        return None

def save_checkpoint(checkpoint_dir,device,model,iteration=0):
    model.eval()
    
    #pytroch saves gpu id into the state dict, 
    #so if model is on gpu, put it to cpu for saving
    if not str(device)=="cpu":
        model.cpu()

    checkpoint = {}
    checkpoint["model"] = model.state_dict()
    checkpoint["iteration"] = iteration
    print("saving model....")
    save_path = os.path.join(checkpoint_dir,"checkpoint_iter%d.pth.tar"%iteration)
    torch.save(checkpoint, save_path)
    print("model saved at",save_path)

    if not str(device)=="cpu":
        #put it back to original device
        model.to(device)