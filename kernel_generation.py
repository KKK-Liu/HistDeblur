import os
import numpy as np
from itertools import product

def real_f(rou,r,z):
    from scipy import special
    lam = 500*10**(-6)
    k=2*np.pi/lam
    NA = 1.5
    ni = 1.5
    
    return special.j0(k*NA*r*rou/ni)*np.cos(0.5*k*rou*rou*z*NA*NA/ni/ni)*rou

def image_f(rou,r,z):
    from scipy import special
    lam = 500*10**(-6)
    k=2*np.pi/lam
    NA = 1.5
    ni = 1.5
    
    return special.j0(k*NA*r*rou/ni)*np.sin(0.5*k*rou*rou*z*NA*NA/ni/ni)*rou
    

def real_part(r,z):
    from scipy import integrate
    
    return integrate.tplquad(real_f,0,1,0,r,0,z)[0]

    
def image_part(r,z):
    from scipy import integrate
    
    return integrate.tplquad(image_f,0,1,0,r,0,z)[0]

def hrz(r,z):
    return np.sqrt(np.power(real_part(r,z),2)+np.power(image_part(r,z),2))
    

def hxzy(x,y,z, lam):
    from scipy import integrate
    from scipy import special
    
    r = np.sqrt(np.power(x,2)+np.power(y,2))
    
    k = 2*np.pi/lam
    NA = 1.4
    ni = 1.5
            
    def hrz():
        def real_f(rou):
            return special.j0(k*NA*r*rou/ni)*np.cos(0.5*k*rou*rou*z*NA*NA/ni/ni)*rou

        def image_f(rou):
            return special.j0(k*NA*r*rou/ni)*np.sin(0.5*k*rou*rou*z*NA*NA/ni/ni)*rou 
        
        def real_part():
            return integrate.quad(real_f,0,1)[0]
        
        def image_part():
            return integrate.quad(image_f,0,1)[0]
        
        if z == 0:
            return np.power(real_part(),2)
        else:
            return np.power(real_part(),2)+np.power(image_part(),2)
    
    return hrz()


def calculate(name,batch_data):
        print(name,' start')
        save_root = './plots_npy/color-real-3um/'
        low = -5.25 * 10 **(-6)
        high = 5.25 * 10 **(-6)
        num = 101
        x = np.linspace(low,high,num) 
        y = np.linspace(low,high,num) 
        
        for item, (lam, name) in batch_data:
            lam = lam * 10 ** (-9)
            item_z = item * 10 **(-7)
            z = np.ones(num) * item_z
            zz = np.zeros((num,num))
            
            for i,j in product(range(num), range(num)):
                zz[i,j] = hxzy(x[i],y[j],z[i], lam)

            file_name = os.path.join(save_root, '{}-{:0>2}.npy'.format(name, int(item)))
            
            np.save(file_name, zz)
            print(file_name)
        print(name,' finish')
        
def psf_map_generate_zxy_positivez_real_multi_thread():
    from itertools import product
    from multiprocessing import Process
    # from threading import Thread
    NUM_OF_THREAD =12
    
    save_root = './plots_npy/color-real-3um/'
    os.makedirs(save_root,exist_ok=True)
    
    print('make diretory')
    
    low = -5.25 * 10 **(-6)
    high = 5.25 * 10 **(-6)
    num = 101
    x = np.linspace(low,high,num) 
    y = np.linspace(low,high,num) 
    # R:700.0, G:546.1, B:435.8 nm 
    all_items = list(product(np.linspace(0,63,64),zip([700, 546.1, 435.8], ['R', 'G', 'B'])))
    
    step = int(len(all_items)/NUM_OF_THREAD)
    
    all_items_batches = [all_items[i:i+step] for i in range(0,len(all_items), step)]
    print(len(all_items_batches),' batches')
    
    ts = [Process(target=calculate, args=(i, all_items_batches[i],)) for i in range(NUM_OF_THREAD)]
    
    [t.start() for t in ts]
    [t.join() for t in ts]
    


if __name__ == '__main__':
    ...
