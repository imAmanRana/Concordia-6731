package application.facerecognizer;
import org.bytedeco.javacpp.opencv_core;

public class Point{
    int x;
    int y;

    Point(int x, int y){
        this.x = x;
        this.y = y;
    }

    public static Point centre(opencv_core.Rect rect){
        return new Point(rect.x()+rect.width()/2, rect.y()+rect.height()/2);
    }

    public static Point centre(Point p1, Point p2){
        return new Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
    }

    public static int x_distance(Point p1, Point p2){
        return Math.abs(p1.x - p2.x);
    }

    public static int y_distance(Point p1, Point p2){
        return Math.abs(p1.y - p2.y);
    }

    public static int x_midpoint(Point p1, Point p2){
        return (p1.x + p2.x)/2;
    }

    public static int y_midpoint(Point p1, Point p2){
        return (p1.y + p2.y)/2;
    }

    public static double angle1(Point p1, Point p2){
        return Math.atan((double) Point.y_distance(p1, p2) / (double) Point.x_distance(p1, p2));
    }

    public static double angle2(Point p1, Point p2){
        return Math.atan((double) (p1.y - p2.y) / (double) (p1.x - p2.x));
    }

    public void rotate(double radians){
        int newX = (int) (Math.cos(radians) * this.x - Math.sin(radians) * this.y);
        int newY = (int) (Math.sin(radians) * this.x + Math.cos(radians) * this.y);

        this.x = newX;
        this.y = newY;
    }

    public void rotate(double radians, Point centre){
        //translate centre to origin
        this.x -= centre.x;
        this.y -= centre.y;

        //rotate
        this.rotate(radians);

        //translate centre back
        this.x += centre.x;
        this.y += centre.y;
    }

    public opencv_core.Point2f toPoint2f(){
        return new opencv_core.Point2f(this.x, this.y);
    }

    public String toString(){
        return "(" + this.x + ", " + this.y + ")";
    }
}
