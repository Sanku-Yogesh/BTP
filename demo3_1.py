import numpy as np
from scipy import linalg as splinalg
from scipy.sparse import csr_matrix
import time
import cv2
import multiprocessing

def GoDec(X,rank,card,power,stop):
    ###########################################################################
    #                        GoDec Algotithm
    ###########################################################################
    #INPUTS:
    #X: nxp data matrix with n samples and p features
    #rank: rank(L)<=rank
    #card: card(S)<=card
    #power: >=0, power scheme modification, increasing it lead to better
    #accuracy and more time cost
    #OUTPUTS:
    #L:Low-rank part
    #S:Sparse part
    #RMSE: error
    #error: ||X-L-S||/||X||
    ###########################################################################
    #REFERENCE:
    #Tianyi Zhou and Dacheng Tao, "GoDec: Randomized Lo-rank & Sparse Matrix
    #Decomposition in Noisy Case", ICML 2011
    ###########################################################################
    #Tianyi Zhou, 2011, All rights reserved.

    #iteration parameters
    iter_max=1e+2;
    error_bound=1e-3;
    iteration=1;
    RMSE=[];

    #matrix size
    [m,n]=X.shape;
    n1 = n;
    if(m<n):
        X=X.T
        n1 = m

    #initialization of L and S
    L=X.copy();
    S=csr_matrix(X.shape);

    t = time.time();
    while(True and (stop>0)):

        #Update of L
        Y2=np.random.randn(n1,rank);
        for i in range(1,power+2):
            Y1=L@Y2;
            Y2=L.T@Y1;

        [Q,R]=splinalg.qr(Y2,mode='economic');
        L_new=(L@Q)@Q.T;

        #Update of S
        T=L-L_new+S;
        L=L_new.copy();

        i = (-abs(T)).argsort(axis=None, kind='mergesort')
        i = i[0:card]
        j = np.unravel_index(i, T.shape)
        indices = np.vstack(j).T

        S=np.zeros(X.shape);

        for idx in indices:
            S[idx[0],idx[1]]=T[idx[0],idx[1]];

        #Error, stopping criteria

        for idx in indices:
            T[idx[0],idx[1]]=0;
        RMSE=[RMSE , np.linalg.norm(T.flatten('F'))];

        if(RMSE[-1]<error_bound or iteration>iter_max):
            break;
        else:
            L=L+T;
        iteration=iteration+1;

    elapsed = time.time() - t;
    LS=L+S;

#     print(X.flatten('F')[100:110])
#     print(np.linalg.norm(X.flatten('F')))
    error=np.linalg.norm(LS.flatten('F')-X.flatten('F'))/np.linalg.norm(X.flatten('F'));
    if(m<n):
        LS=LS.T;
        L=L.T;
        S=S.T;
    return L,S,RMSE,error

def captureframes(v,displayframes,displaylock,filepath,frames,frameslock):
    if filepath==0:
        video_obj = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        skip = 0
    else:
        video_obj = cv2.VideoCapture(filepath)
        skip = 14
#         skip = 9
    fps = 30.0
#     fps = 20.0
    
    skippedFrames = 0;
    print("capture")
    timeStamp = time.time()
    while(v.value>0):
        for j in range(0,skip+1):
            isframe,frame = video_obj.read()
            with displaylock:
                if(not displayframes.empty()):
                    displayframes.get()
                    skippedFrames = skippedFrames + 1
                displayframes.put(frame)
            if filepath!=0:
                time.sleep(0.35/fps)
        if(not isframe):
            v.value = 0
            break
        with frameslock:
            if(frames.full()):
                frames.get()
            frames.put(frame)
        #v.value = v.value-1
    print(" :( Avg number of Skipped Frames per Second : " + str(skippedFrames/(time.time()-timeStamp)))
    print("captureframes ended")
        
def processframes(v,frames,processedframes,frameslock,processedframeslock):
    r = 256 ; c = 320;
    print("process")
    while v.value:
        frameslist = []
        processedframeslist = []
        with frameslock:
            while(not frames.empty()):
                frameslist.append(frames.get().copy())
        for frame in frameslist:
            grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scaledframe = cv2.resize(grayframe, (c,r)) #note
            scaledframe = scaledframe.astype(np.float32)
            floatscaledframe = scaledframe/255.0
            processedframeslist.append(floatscaledframe.flatten('F').copy())
        
        for temp in processedframeslist:
            with processedframeslock:
                if(processedframes.full()):
                    processedframes.get()
                processedframes.put(temp.copy())
    print("processframes ended")

                
def estimatebg(v,processedframes,bg,processedframeslock,bglock):
    numberFrames = 200;
    r = 256 ; c = 320; 
    X = np.zeros([r*c,numberFrames],dtype = np.float32)
    rank = 2;
    card = 310000;
    power = 0;
    j = 1;
    print("estimate")
    while v.value:
        with processedframeslock:
            print("processedframes size  ",processedframes.qsize())
            if(j>0):
                i = 0;
            else:
                i = numberFrames-1;
            while(not processedframes.empty()):
                X[:,i] = processedframes.get().copy()
                if(j>0):
                    i = i+1;
                else:
                    i = i-1;
        j = -1*j
        print("godec started")
        [L,S,RMSE,error]=GoDec(X.copy(),rank,card,power,v.value);
        print(L[100:110])
        print("godec ended")
        with bglock:
            if(not bg.empty()):
                bg.get()
            y = np.matrix(L.copy())
            y = y.mean(1)
            bg.put(y[:,0].copy())
#             bg.put(L[:,numberFrames-1].copy())
#             bg.put(X[:,numberFrames-1]-S[:,numberFrames-1])
            
    print("estimatebg ended")

            
def display(v,displayframes,displaylock,filepath,bg,bglock):
    
    numberFrames = 200
    r = 256 ; c = 320;
    fps = 30.0;
#     fps = 20.0
    timeStamps = [];
    frameChanged = 0;
    
    noOfFrames = 0;
    
    background = np.zeros([r*c,numberFrames],dtype = np.float32)
    
    writerBB = cv2.VideoWriter('BoundingBoxVideo.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (c, r),True)
    writerSP = cv2.VideoWriter('SparsePartVideo.avi',cv2.VideoWriter_fourcc(*'MJPG'),fps,(c,r),False)
    
    print("display")
    
    while(displayframes.empty()):
        time.sleep(1)
    
    timeStampDisplay = time.time();
    while v.value:
        with displaylock:
            if(not displayframes.empty()):
                frame = displayframes.get()
                frameChanged = 1
        if(frameChanged):
            grayframe  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scaledframe = cv2.resize(grayframe, (c,r)) #note
            frame1 = cv2.resize(frame,(c,r))           #note
            scaledframe = scaledframe.astype(np.float32)
            floatscaledframe = scaledframe/255.0
            with bglock:
                if(not bg.empty()):
                    background = bg.get().copy()

            temp = np.resize(background, (c,r));
            temp = temp.T
            sparse = 255.0*(abs(floatscaledframe-temp))
            sparse1 = sparse.astype('uint8')

            blur = cv2.GaussianBlur(sparse1, (5,5), 0)

            _, thresh = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)

            SE = np.ones((9,9),np.uint8)
            closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, SE)
            contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                if cv2.contourArea(contour) < 200 or float(w)/h > 5 or float(h)/w>5:
                    continue
                cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

            displayframe = cv2.resize(frame1, (960,540))
            displaysparse = cv2.resize(sparse1,(960,540))

            noOfFrames = noOfFrames + 1
            
            cv2.imshow("tracking",displayframe)
            cv2.imshow("sparse",displaysparse)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                v.value = 0
            writerBB.write(frame1)
            writerSP.write(sparse1)
            frameChanged = 0
    print(" :) Avg fps of the video is : " + str(noOfFrames/(time.time()-timeStampDisplay)))
    print("display ended")
    
if __name__ == '__main__':

    print("entered")
    
    filepath = './inputVideos/'+'5to14_2.mp4'
    maxsize = 201
    noframes = 200
    
    v = multiprocessing.Value('i',401)
    frames = multiprocessing.Queue(maxsize)
    processedframes = multiprocessing.Queue(noframes)
    bg = multiprocessing.Queue()
    displayframes = multiprocessing.Queue()
    
    if(v.value == 401):
        if filepath==0:
            video_obj = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            skip = 0
        else:
            video_obj = cv2.VideoCapture(filepath)
            skip = 14
#             skip = 9
        numberFrames = 200;
        r = 256 ; c = 320;
        #r = 1080 ; c = 1920;
        X = np.zeros([r*c,numberFrames],dtype = np.float32)
        frameslist = []
        
        for i in range(0,numberFrames):
            for j in range(0,skip):
                isframe,frame = video_obj.read()
            isframe,frame = video_obj.read()
            if(not isframe):
                break
            frameslist.append(frame)
            grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scaledframe = cv2.resize(grayframe, (c,r)) #note
            scaledframe = scaledframe.astype(np.float32)
            floatscaledframe = scaledframe/255.0
            X[:,i] = floatscaledframe.flatten('F')
        video_obj.release()
        rank = 2;
        card = 310000;
        power = 0;
        [L,S,RMSE,error]=GoDec(X.copy(),rank,card,power,1);
    
    v.value = 400
    
    y = np.matrix(L.copy())
    y = y.mean(1)
    bg.put(y[:,0].copy())
#     bg.put(L[:,numberFrames-1].copy())
#     bg.put(X[:,numberFrames-1]-S[:,numberFrames-1])
    
    for i in range(0,numberFrames):
        processedframes.put(X[:,i].copy())
        frames.put(frameslist[i].copy())
    
    frameslock = multiprocessing.Lock()
    processedframeslock = multiprocessing.Lock()
    bglock = multiprocessing.Lock()
    displaylock = multiprocessing.Lock()
    
    print("starting process")
    
    capturingframes = multiprocessing.Process(name="capture",target = captureframes, args=(v,displayframes,displaylock,filepath,frames,frameslock,))
    
    processingframes = multiprocessing.Process(name="process",target = processframes, args=(v,frames,processedframes,frameslock,processedframeslock,))
    
    estimatingbg = multiprocessing.Process(name="estimate",target = estimatebg, args=(v,processedframes,bg,processedframeslock,bglock,))
    
    displaying = multiprocessing.Process(name="display",target = display, args=(v,displayframes,displaylock,filepath,bg,bglock,))
    
    capturingframes.start()
    time.sleep(1)
    processingframes.start()
    time.sleep(1)
    estimatingbg.start()
    time.sleep(1)
    displaying.start()
    
    while(v.value>0):
        time.sleep(1)
    time.sleep(5)
    capturingframes.terminate()
    processingframes.terminate()
    estimatingbg.terminate()
    displaying.terminate()