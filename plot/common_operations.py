import os, sys
sys.path.append(os.getcwd())

def savefig(plt, path, show_message= True, tight_flag= True, pad_inches= 0, newline= True):
    if show_message:
        print("Saving to {}".format(path))
    if tight_flag:
        plt.savefig(path, bbox_inches='tight', pad_inches= pad_inches)
    else:
        plt.savefig(path)
    if newline:
        print("")
