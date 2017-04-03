import numpy as np
import math

class point:
    def __init__(self, x, y, name):
        self.x=x
        self.y=y
        self.name=name

    cluster=None

    def assign_cluster(self, this_cluster):
        old_cluster=self.cluster
        self.cluster=this_cluster

###############################
class cluster:
    def __init__(self, x, y, name):
        self.x=x
        self.y=y
        self.name=name
    points=[]
##
##    def add_points(self,this_point):
##        self.points.append(this_point)
##        this_point.assign_cluster(self)

    def update_centroid(self, points):
        xtotal=0.0
        ytotal=0.0
        cluster_points=[]

        for point in points:
            if point.cluster==self:
                cluster_points.append(point)

        if len(cluster_points)>0:
            for point in cluster_points:
                xtotal=xtotal+point.x
                ytotal=ytotal+point.y

            x_bar=xtotal/len(cluster_points)
            y_bar=ytotal/len(cluster_points)
            self.x=x_bar
            self.y=y_bar


###############################
def dist(p1,p2):
    d=math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    return np.round(d,3)


###############################
def assign_single_point(point, clusters):
    dist_to_cluster=[]
    p2c_dist=1000000.0
    for cluster in clusters:
        p2c_dist=dist((point.x,point.y),(cluster.x, cluster.y))
        dist_to_cluster.append(p2c_dist)
    point.assign_cluster(clusters[np.argmin(np.array(dist_to_cluster))])
    return point


###############################
def update_clusters(points, clusters):
    for cluster in clusters:
        cluster.update_centroid(points)
    return clusters



###############################
def assign_points(points, clusters):
    dist_to_cluster=[]
    for point in points:
        point=assign_single_point(point, clusters)

    display_centroid(points, clusters)
    clusters=update_clusters(points, clusters)


    return points, clusters



###############################
def display_centroid(points, clusters):
    pstr=""
    for cluster in clusters:
        pstr=""
        x=str(np.round(cluster.x,1))
        y=str(np.round(cluster.y,1))

        for point in points:
            if point.cluster==cluster:
                pstr=pstr+point.name+" "

        print "centroid for %s is %s with %s" %(cluster.name, (x,y), pstr)

###############################
def display_point_centroid(point, clusters):
    dist_str=""
    for cluster in clusters:
        dist_value=dist((point.x, point.y),(cluster.x, cluster.y))


###############################
if __name__ == '__main__':

## W5B Q1

##    gp1=point(25.0,125.0, 'green1')
##    gp2=point(44.0,105.0, 'green2')
##    gp3=point(29.0, 97.0, 'green3')
##    gp4=point(35.0, 63.0, 'green4')
##    gp5=point(55.0, 63.0, 'green5')
##    gp6=point(42.0, 57.0, 'green6')
##    gp7=point(23.0, 40.0, 'green7')
##    gp8=point(64.0, 37.0, 'green8')
##    gp9=point(33.0, 22.0, 'green9')
##    gp10=point(55.0, 20.0, 'green10')
##
##    gl1=point(50.0, 30.0, 'gold1')
##    gl2=point(50.0, 60.0, 'gold2')
##    gl3=point(43.0, 83.0, 'gold3')
##    gl4=point(50.0, 90.0, 'gold4')
##    gl5=point(63.0, 88.0, 'gold5')
##    gl6=point(38.0, 115.0, 'gold6')
##    gl7=point(55.0, 118.0, 'gold7')
##    gl8=point(50.0, 130.0, 'gold8')
##    gl9=point(28.0, 145.0, 'gold9')
##    gl10=point(65.0, 140.0, 'gold10')
##
##    c1=cluster(gp1.x, gp1.y, 'c1')
##    c2=cluster(gp2.x, gp2.y, 'c2')
##    c3=cluster(gp3.x, gp3.y, 'c3')
##    c4=cluster(gp4.x, gp4.y, 'c4')
##    c5=cluster(gp5.x, gp5.y, 'c5')
##    c6=cluster(gp6.x, gp6.y, 'c6')
##    c7=cluster(gp7.x, gp7.y, 'c7')
##    c8=cluster(gp8.x, gp8.y, 'c8')
##    c9=cluster(gp9.x, gp9.y, 'c9')
##    c10=cluster(gp10.x, gp10.y, 'c10')
##
##    gp1.assign_cluster(c1)
##    gp2.assign_cluster(c2)
##    gp3.assign_cluster(c3)
##    gp4.assign_cluster(c4)
##    gp5.assign_cluster(c5)
##    gp6.assign_cluster(c6)
##    gp7.assign_cluster(c7)
##    gp8.assign_cluster(c8)
##    gp9.assign_cluster(c9)
##    gp10.assign_cluster(c10)
##
##
##    points=[gp1, gp2, gp3, gp4, gp5, gp6, gp7, gp8, gp9, gp10, \
##            gl1, gl2, gl3, gl4, gl5, gl6, gl7, gl8, gl9, gl10]
##
##
####    points=[gl1, gl2, gl3, gl4, gl5, gl6, gl7, gl7, gl9, gl10]
##
##    clusters=[c1,c2,c3,c4,c5,c6,c7,c8,c9,c10]
##
####    print "original state"
####    display_centroid(points, clusters)
##
##
##    print "state 1"
##    points, clusters=assign_points(points, clusters)
##
##    print "state 2"
##    points, clusters=assign_points(points, clusters)

##    points, clusters=assign_points(points, clusters)
##    display_centroid(points, clusters)



## W5B Q2
    c1=cluster(5,10, "c1")
    c2=cluster(20,5, "c2")
    clusters=[c1, c2]

##  1
    yul=point(3,3,"yellow_ul")
    ylr=point(10,1, "yellow_lr")
    bul=point(15,14, "blue_ul")
    blr=point(20,10, "blue_lr")


##  2 - false
##    yul=point(7,8,"yellow_ul")
##    ylr=point(12,5, "yellow_lr")
##    bul=point(13,10, "blue_ul")
##    blr=point(16,4, "blue_lr")

##  3 -false
##    yul=point(6,15,"yellow_ul")
##    ylr=point(13,7, "yellow_lr")
##    bul=point(16,16, "blue_ul")
##    blr=point(18,5, "blue_lr")

##  4- false
##    yul=point(3,3,"yellow_ul")
##    ylr=point(10,1, "yellow_lr")
##    bul=point(13,10, "blue_ul")
##    blr=point(16,4, "blue_lr")


    points=[yul,ylr,bul,blr]

    print "state 1"
    points, clusters=assign_points(points, clusters)

    print "state 1"
    points, clusters=assign_points(points, clusters)





