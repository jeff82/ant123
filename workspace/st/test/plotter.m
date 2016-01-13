close all
clc
clear all
dot=load('E:\Documents\py\ss.txt');%('dot2.txt');%
x=dot(:,1);
y=dot(:,2);
%plot(x,y)
atx=find(x<1);
atl=length(atx)
xx(1,:)=x(1:atx(1)-1)
yy(1,:)=y(1:atx(1)-1)

hold on
pause(1)
axis([635 685 275 310]);
plot(xx(1,:),yy(1,:))
lls=0
for ll=1:atl-1
    xxd=x(atx(ll)+1:atx(ll+1)-1);
    yyd=y(atx(ll)+1:atx(ll+1)-1);
    plot( xxd, yyd,'r-')
    plot(xxd(length(xxd)),yyd(length(yyd)),'*')
    pause(0.6)
    if ll<atl-1
        plot( [x(atx(ll+1)-1),x(atx(ll+1)+1)], [y(atx(ll+1)-1),y(atx(ll+1)+1)],'g-.')
        lls=sqrt((x(atx(ll+1)-1)-x(atx(ll+1)+1))^2+(y(atx(ll+1)-1)-y(atx(ll+1)+1))^2)+lls;
    end
    pause(0.01)
end
lls
