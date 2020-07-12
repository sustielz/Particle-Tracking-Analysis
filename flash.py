import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['figure.dpi'] = 200
import cv2, json


#### Load intensity as a json file
def load_intensity(filename, path='', thresh=0.5):  ## Load data from json file
    filename = filename.split('.')[0]  ## Precaution, in case filename includes suffix
    if path is not '' and path[-1] is not '/':
        path = path + '/'
    with open(path+'Intensity/'+filename+'.json', 'r') as f:
        preds =  json.load(f)
    return np.array(preds)


def filt_d1(led, thresh=0.7):
    N = np.size(led)
    ip = np.arange(N) + 1
    ip[N-1] = N-2
    im = np.arange(N) - 1
    im[0] = 1
    led = np.array(led)
    base = (led[ip] + led[im])/2
    return (led - base)/2

# def filt_d2(preds, thresh=0.7):
#     N = len(preds)
#     ip = np.arange(N) + 1
#     ip[N-1] = N-2
#     ipp = ipp + 1
    
#     im = np.arange(N) - 1
#     im[0] = 1
#     preds = np.array(preds)
#     base = (preds[ip] + preds[im])/2
#     return (preds - base)

 


######## Code to Test Filtering/Detection Methods ########
def test_filter(led, filt=filt_d1, thresh=0.7):
    f1 = led - np.mean(led)
    f2 = filt(led)
    f3 = detect(led, thresh=thresh)
    f4 = detect(f2, thresh=thresh)

    for i in range(np.size(led)):
        if f4[i]:
            plt.axvline(x=i+0.5, color='orange')
        if f3[i]:
            plt.axvline(x=i-0.5, color='blue')


    plt.plot(f1, label='Raw Intensity (mean subtracted)')
    plt.plot(f2, label='Filtered Intensity')    
    plt.legend(prop={'size': 7})

    plt.xlabel('Frame')
    plt.title('Detecting LED Flashes (threshold={:.2f})'.format(thresh))


    
#### Detect flashes from intensity; return boolean array
def detect(led, thresh=0.7):  
    thresh = min(led) + thresh*( max(led) - min(led) )
    fl = np.zeros(np.size(led), dtype=bool)
    fl[np.array(led)>thresh] = True
    return fl



def intersect(i, j, f=6.18, dt=20/1000, FPS=30):
    dT = abs(j - i)/FPS
    high = 0
    while dT > high:
        high += 1/f
        
    if high - dT < dt:
        return [i/FPS - dt, i/FPS - (high - dT)]
    elif dT - (high-1/f) < dt:
        return [i/FPS + dT - (high-1/f) - dt, i/FPS]    
    
def intersect2(i, j, f=6.18, dt=20/1000, FPS=30):
    dT = abs(j - i)/FPS
    high = 0
    while dT > high:
        high += 1/f
        
    if high - dT < dt:
        return [i/FPS - dt, i/FPS - (high - dT)]
    elif dT - (high-1/f) < dt:
        return [i/FPS + dT - (high-1/f) - dt, i/FPS]    
    
    

def Intersect(ivl, j, f=6.18, dt=20/1000, FPS=30):
    lows = []
    highs = []
    ivls = []
    for j in jvals:
        ivl = intersect(i, j, f=f, dt=dt, FPS=FPS)
        lows.append(ivl[0])
        highs.append(ivl[1])
        ivls.append(ivl)
        
#### This function analyzes where the flashes are, and narrows the interval where the zero crossing can lie. 
#### Arguments to pass are:    frequency f,      flash duration dt,       and framerate FPS
def INTERSECT(FL, f=6.18, dt=20/1000, FPS=30):
#     ivls = []
#     for i in FL:
#         for j in FL:
#             ivl = intersect(i, j, f=f, dt=dt, FPS=FPS)
#             ivls.append(ivl - i/FPS)
    ivls = [intersect(i, j, f=f, dt=dt, FPS=FPS) - i/FPS for i in FL for j in FL]
    lows = [ivl[0] for ivl in ivls]
    highs = [ivl[1] for ivl in ivls]

    return [max(lows), min(highs)]

def report_ivl(ivl, FL=[], dt=20/1000, FPS=30):
    print('Interval narrowed from {} ({} Hz res.) to {} ({:.5f} Hz res.)'.format([-dt, 0], 1/dt, ivl, 1/(ivl[1] - ivl[0])))
    for fl in FL:
        print('Frame {} (t={:.2f}), {}'.format(fl, fl/FPS, ivl + fl/FPS))

