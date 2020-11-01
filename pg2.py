import cv2
import colorsys as cs
import numpy as np

opener=7
def f(filename):
    image=cv2.imread(filename,1)
    img=contrast(image)
    
    m=len(img)
    n=len(img[0])
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,20,200)

    kernel1 = np.ones((1,opener),np.uint8)
    th1 = cv2.morphologyEx(edges,cv2.MORPH_OPEN,kernel1)
    edges1 = cv2.Canny(th1,100,200)

    minLineLength = 10
    maxLineGap = 2
    y=[]
    linesh = cv2.HoughLinesP(edges1,1,np.pi/180,1,minLineLength,maxLineGap)
    for x1,y1,x2,y2 in linesh[0]:
        y.append((y1+y2)//2)
    y=aggregate(y,2)
    clean(y)
    if(len(y)<8):
        y=[]
        linesh = cv2.HoughLinesP(edges1,1,np.pi/180,1,minLineLength,maxLineGap)
        for x1,y1,x2,y2 in linesh[0]:
            y.append((y1+y2)//2)
        y=aggregate(y,1)
        clean(y)
        
    ymax=max(y)
    ymin=min(y)
                
    kernel2 = np.ones((opener,1),np.uint8)
    th2 = cv2.morphologyEx(edges,cv2.MORPH_OPEN,kernel2)
    edges2 = cv2.Canny(th2,100,200)

    minLineLength = 100
    maxLineGap = 5
    
    x=[]
    linesv = cv2.HoughLinesP(edges2,1,np.pi/180,1,minLineLength,maxLineGap)
    for x1,y1,x2,y2 in linesv[0]:
        if ymin<y1<ymax:
            if ymin<y2<ymax:
                x.append((x1+x2)//2)
                
    x=aggregate(x,5)
    
    l_x=max(x)-min(x)
    
    xmin=min(x)+l_x//6
    xmax=max(x)-l_x//6
    
    for i in y:
        cv2.line(image,(xmin,i),(xmax,i),(0,0,0),2)
    cv2.line(image,(xmin,ymin),(xmin,ymax),(0,0,0),2)
    cv2.line(image,(xmax,ymin),(xmax,ymax),(0,0,0),2)
    
    #-- Fin de segmentation --#
    
    colors=[]
    for i in range(len(y)-1):
        colors.append(get_mean_color(image,xmin+1,xmax-1,y[i]+1,y[i+1]-1,i))
    colors_str=colors_interpret(colors)
    colors_val=colors_translate(colors_str)
    sign_val=0
    ncolors=len(colors_val)
    for i in range(ncolors-2):
        sign_val=10*sign_val+colors_val[i]
    print(`sign_val*10**colors_val[ncolors-2]`+' +/-'+`colors_val[ncolors-1]`+'% Ohms')
    cv2.imshow(filename,image)
        


def contrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return(final)


def aggregate(y,s=3):
    y.sort()
    l=len(y)
    grad=[]
    for i in range(l-1):
        grad.append(y[i+1]-y[i])
    mean=np.mean(grad)
    ytmp=[]
    tmp=[y[0]]
    for i in range(l-1):
        if grad[i]<mean+s:
            tmp.append(y[i+1])
        else:
            if len(tmp)>0:
                ytmp.append(tmp)
                tmp=[y[i+1]]
    if len(tmp)>0:
        ytmp.append(tmp)

    y_final=[]
    for i in range(len(ytmp)):
        if len(ytmp[i])>s:
            y_final.append(int(np.mean(ytmp[i])))
    return(y_final)

def clean(y):
    y.sort()
    grad=[]
    for i in range(len(y)-1):
        grad.append(y[i+1]-y[i])
    m=np.mean(grad)
    std=np.std(grad)
    if grad[0]>=m+std:
        y.pop(0)
    if grad[-1]>=m+std:
        y.pop(-1)

def get_mean_color(img,x1,x2,y1,y2,b):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    r=0
    g=0
    b=0
    k=0
    for j in range(x1,x2,1):
        for i in range(y1,y2,1):
            if hsv[i,j,2]<220:
                k=k+1
                b=b+img[i,j,0]
                g=g+img[i,j,1]
                r=r+img[i,j,2]
    c_hsv=cs.rgb_to_hsv(float(r)/k/255,float(g)/k/255,float(b)/k/255)
    return([c_hsv[0]*360,c_hsv[1]*100,c_hsv[2]*100])

def colors_interpret(colors):
    ci=[]
    for i in range(len(colors)):
        ci.append(color_det(colors[i]))
    if ci[0]=='':
        ci[0]='Or'
        ci.reverse()
    elif ci[-1]=='':
        ci[-1]='Or'
    cif=[]
    for i in ci:
        if i!='':
            cif.append(i)
    return(cif)

def color_det(c):
    if c[1]<15 and c[2]<50:
        return('Noir')
    elif c[1]<15 and c[2]>=50:
        return('Blanc')
    elif 270<=c[0]<320:
        return('Violet')
    elif (320<=c[0]<360 or 0<=c[0]<13) and c[1]+c[2]>120:
        return('Rouge')
    elif (320<=c[0]<360 or 0<=c[0]<13) and c[1]+c[2]<=120:
        return('Marron')
    elif 45<=c[0]<75 and c[1]>70:
        return('Jaune')
    elif 13<=c[0]<45 and c[1]>40 and c[2]>75:
        return('Orange')
    elif 110<=c[0]<140:
        return('Vert')
    else:
        return('')

def colors_translate(c,k=2):
    n=len(c)
    c_values=[]
    for i in range (n):
        c_values.append(color_translate(c[i],i==n-1,i==n-2))
    if min(c_values)==-1 and k>0:
        return(colors_translate(c.reverse(),k-1))
    else:
        return(c_values)
        

def color_translate(c,tolerance,multiplier):
    if c=='Noir':
        if tolerance==1:
            s=-1
        else:
            s=0
    elif c=='Marron':
        s=1
    elif c=='Rouge':
        s=2
    elif c=='Orange':
        if tolerance==1:
            s=-1
        else:
            s=3
    elif c=='Jaune':
        if tolerance==1:
            s=-1
        else:
            s=4
    elif c=='Vert':
        if tolerance==1:
            s=0.5
        else:
            s=5
    elif c=='Bleu':
        if tolerance==1:
            s=0.25
        else:
            s=6
    elif c=='Violet':
        if tolerance==1:
            s=0.10
        else:
            s=7
    elif c=='Gris':
        if tolerance==1:
            s=0.05
        else:
            s=8
    elif c=='Blanc':
        if tolerance==1:
            s=-1
        else:
            s=9
    elif c=='Or':
        if tolerance==1:
            s=5
        elif multiplier==1:
            s=0.1
        else:
            s=-1
    elif c=='Argent':
        if tolerance==1:
            s=10
        elif multiplier==1:
            s=0.01
        else:
            s=-1
    return(s)
    
        

#def second_argmin(y):
#    argmin1=np.argmin(y)
#    argmin2=[np.argmax(y)]
#    for i in range(len(y)):
#        if i!=argmin1:
#            if y[i]==y[argmin2[0]]:
#                argmin2.append(i)
#            if y[argmin1]<=y[i]<y[argmin2[0]]:
#                argmin2=[i]
#    return(argmin1,argmin2)
#
f('img.png')
#print(" ")
#f('img_1_resized.png')
#print(" ")
#f('img_2_resized.png')
#print(" ")
#f('img_3_resized.png')
#print(" ")
#f('img_4_resized.png')
#print(" ")
#f('img_5_resized.png')
#print(" ")
#f('img_6_resized.png')


cv2.waitKey(0)
cv2.destroyAllWindows()