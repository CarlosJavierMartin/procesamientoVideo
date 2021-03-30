import processing.video.*;
import cvimage.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

Capture cam;
CVImage img,auximg, miximg, mixauximg, specialimg, specialauximg;

int normalVal = 1, xCoor = 0;
boolean menuClose = false;

void setup(){
  size(1280, 640);  
  cam = new Capture(this, width, height);
  cam.start();

  //OpenCV
  //Carga biblioteca core de OpenCV
  System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
  println(Core.VERSION);
  //Crea imágenes
  img = new CVImage(cam.width, cam.height);
  auximg = new CVImage(cam.width, cam.height);
  miximg = new CVImage(cam.width/2, cam.height);
  mixauximg = new CVImage(cam.width/2, cam.height);
  specialimg = new CVImage(cam.width, cam.height);
  specialauximg = new CVImage(400, cam.height);
}

void draw(){
  if (!menuClose){
    showMenu();
  } else {
    if (cam.available()){
      switch (normalVal){
        case 1:
        normalMode();
        break;
        case 2:
        greyMode(false);
        break;
        case 3:
        umbralMode(false);
        break;
        case 4:
        edgeMode(false);
        break;
        case 5:
        greyMode(true);
        break;
        case 6:
        umbralMode(true);
        break;
        case 7:
        edgeMode(true);
        break;
        case 8:
        greyEdgeMode();
        break;
        case 9:
        edgeGreyMode();
        break;
        case 0:
        normalEdgeMode();
        break;
      }
    }
  }
}

void showMenu(){
  background(0);
  textSize(20);
  text("Pulsa la tencla ENTER para alternar entre este menu y la vision de camara.",5,25);
  text("Cambia el filtro de imagen con los numeros 1, 2, 3, 4, 5, 6, 7",5,50);
  text("1: Modo normal",5,75);
  text("2: Modo escala de grises",5,100);
  text("3: Modo umbralizado (mueve el raton horizontalmente para modificar el valor de umbralizado)",5,125);
  text("4: Modo bordes",5,150);
  text("5: Modo mixto normal-escala de grises",5,175);
  text("6: Modo mixto normal-umbralizado (mueve el raton horizontalmente para modificar el valor de umbralizado)",5,200);
  text("7: Modo mixto normal-bordes",5,225);
  text("Usa las teclas izquieda y derecha para mover el rectangulo de superposicion", 5, 250);
  text("8: Modo superposicion bordes-grises",5,275);
  text("9: Modo superposicion grises-bordes",5,300);
  text("0: Modo superposicion normal-bordes",5,325);

}

void normalMode(){
  background(0);
  cam.read();
  //Obtiene la imagen de la cámara
  img.copy(cam, 0, 0, cam.width, cam.height, 
  0, 0, img.width, img.height);
  image(img,0,0);
}

void greyMode(boolean mix){
  background(0);
  cam.read();
  if (mix){
    miximg.copy(cam, 0, 0, cam.width, cam.height, 
    0, 0, miximg.width, miximg.height);
    miximg.copyTo();
    
    Mat gris = miximg.getGrey();
  
    cpMat2CVImage(gris,mixauximg, cam.width/2, cam.height);
    
    image(miximg, 0, 0);
    image(mixauximg, width/2,0);

    gris.release();
  } else{
    img.copy(cam, 0, 0, cam.width, cam.height, 
    0, 0, img.width, img.height);
    img.copyTo();
    
    Mat gris = img.getGrey();
    cpMat2CVImage(gris,auximg, cam.width, cam.height);
    
    image(auximg,0,0);
        
    gris.release();
  }
}

void umbralMode(boolean mix){
  background(0);
  cam.read();
  if (mix){
    miximg.copy(cam, 0, 0, cam.width, cam.height, 
    0, 0, miximg.width, miximg.height);
    miximg.copyTo();
    
    Mat gris = miximg.getGrey();
    
    Imgproc.threshold(gris,gris,255*mouseX/width,255,Imgproc.THRESH_BINARY);
    
    cpMat2CVImage(gris,mixauximg, cam.width/2, cam.height);
    
    image(miximg, 0, 0);
    image(mixauximg, width/2,0);
    
    gris.release();
  } else{
    img.copy(cam, 0, 0, cam.width, cam.height, 
    0, 0, img.width, img.height);
    img.copyTo();
    
    Mat gris = img.getGrey();
    
    Imgproc.threshold(gris,gris,255*mouseX/width,255,Imgproc.THRESH_BINARY);
    
    cpMat2CVImage(gris,auximg, cam.width, cam.height);
    
    image(auximg,0,0);
    
    gris.release();
  }
}

void edgeMode(boolean mix){
  
  background(0);
  cam.read();
  
  if (mix){
    miximg.copy(cam, 0, 0, cam.width, cam.height, 
    0, 0, miximg.width, miximg.height);
    miximg.copyTo();

    Mat gris = miximg.getGrey();
    
    int scale = 1;
    int delta = 0;
    int ddepth = CvType.CV_16S;
    Mat grad_x = new Mat();
    Mat grad_y = new Mat();
    Mat abs_grad_x = new Mat();
    Mat abs_grad_y = new Mat();
    
    Imgproc.Sobel(gris, grad_x, ddepth, 1, 0);
    Core.convertScaleAbs(grad_x, abs_grad_x);
    
    Imgproc.Sobel(gris, grad_y, ddepth, 0, 1);
    Core.convertScaleAbs(grad_y, abs_grad_y);
    
    Core.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gris);
    
    cpMat2CVImage(gris,mixauximg, cam.width/2, cam.height);

    image(miximg, 0, 0);
    image(mixauximg, width/2,0);

    gris.release();
  } else{
    img.copy(cam, 0, 0, cam.width, cam.height, 
    0, 0, img.width, img.height);
    img.copyTo();

    Mat gris = img.getGrey();
    
    int scale = 1;
    int delta = 0;
    int ddepth = CvType.CV_16S;
    Mat grad_x = new Mat();
    Mat grad_y = new Mat();
    Mat abs_grad_x = new Mat();
    Mat abs_grad_y = new Mat();
    
    Imgproc.Sobel(gris, grad_x, ddepth, 1, 0);
    Core.convertScaleAbs(grad_x, abs_grad_x);
    
    Imgproc.Sobel(gris, grad_y, ddepth, 0, 1);
    Core.convertScaleAbs(grad_y, abs_grad_y);
    
    Core.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gris);

    cpMat2CVImage(gris,auximg, cam.width, cam.height);

    image(auximg,0,0);
    
    gris.release();
  }
}

void greyEdgeMode(){
  background(0);
  cam.read();
  specialimg.copy(cam, 0, 0, cam.width, cam.height, 
  0, 0, specialimg.width, specialimg.height);
  specialimg.copyTo();  
  Mat gris = specialimg.getGrey();
    
  int scale = 1;
  int delta = 0;
  int ddepth = CvType.CV_16S;
  Mat grad_x = new Mat();
  Mat grad_y = new Mat();
  Mat abs_grad_x = new Mat();
  Mat abs_grad_y = new Mat();
  
  Imgproc.Sobel(gris, grad_x, ddepth, 1, 0);
  Core.convertScaleAbs(grad_x, abs_grad_x);
  
  Imgproc.Sobel(gris, grad_y, ddepth, 0, 1);
  Core.convertScaleAbs(grad_y, abs_grad_y);
  
  Core.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gris);

  cpMat2CVImage(gris,specialimg, cam.width, cam.height);

  
  specialauximg.copy(cam, xCoor, 0, 400, cam.height, 
  0, 0, specialauximg.width, specialauximg.height);
  specialauximg.copyTo();
  gris = specialauximg.getGrey();

  cpMat2CVImage(gris,specialauximg, 400, cam.height);
  
  
  
  image(specialimg, 0, 0);
  image(specialauximg, xCoor, 0);
  stroke(255);
  line(xCoor, 0.0, xCoor, height);
  line(xCoor+400, 0.0, xCoor+400, height);

  gris.release();
}

void edgeGreyMode(){
  background(0);
  cam.read();
  specialimg.copy(cam, 0, 0, cam.width, cam.height, 
  0, 0, specialimg.width, specialimg.height);
  specialimg.copyTo();  
  Mat gris = specialimg.getGrey();
  
  gris = specialimg.getGrey();
  cpMat2CVImage(gris,specialimg, cam.width, cam.height);  
    
    
  specialauximg.copy(cam, xCoor, 0, 400, cam.height, 
  0, 0, specialauximg.width, specialauximg.height);
  specialauximg.copyTo();
  gris = specialauximg.getGrey();  
  int scale = 1;
  int delta = 0;
  int ddepth = CvType.CV_16S;
  Mat grad_x = new Mat();
  Mat grad_y = new Mat();
  Mat abs_grad_x = new Mat();
  Mat abs_grad_y = new Mat();
  
  Imgproc.Sobel(gris, grad_x, ddepth, 1, 0);
  Core.convertScaleAbs(grad_x, abs_grad_x);
  
  Imgproc.Sobel(gris, grad_y, ddepth, 0, 1);
  Core.convertScaleAbs(grad_y, abs_grad_y);
  
  Core.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gris);
  cpMat2CVImage(gris,specialauximg, 400, cam.height);
  
  
  
  image(specialimg, 0, 0);
  image(specialauximg, xCoor, 0);
  stroke(255);
  line(xCoor, 0.0, xCoor, height);
  line(xCoor+400, 0.0, xCoor+400, height);

  gris.release();
}

void normalEdgeMode(){
  background(0);
  cam.read();
  specialimg.copy(cam, 0, 0, cam.width, cam.height, 
  0, 0, specialimg.width, specialimg.height);

  
  specialauximg.copy(cam, xCoor, 0, 400, cam.height, 
  0, 0, specialauximg.width, specialauximg.height);
  specialauximg.copyTo();
  Mat gris = specialauximg.getGrey();  
  int scale = 1;
  int delta = 0;
  int ddepth = CvType.CV_16S;
  Mat grad_x = new Mat();
  Mat grad_y = new Mat();
  Mat abs_grad_x = new Mat();
  Mat abs_grad_y = new Mat();
  
  Imgproc.Sobel(gris, grad_x, ddepth, 1, 0);
  Core.convertScaleAbs(grad_x, abs_grad_x);
  
  Imgproc.Sobel(gris, grad_y, ddepth, 0, 1);
  Core.convertScaleAbs(grad_y, abs_grad_y);
  
  Core.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gris);
  cpMat2CVImage(gris,specialauximg, 400, cam.height);
  
  
  
  image(specialimg, 0, 0);
  image(specialauximg, xCoor, 0);
  stroke(255);
  line(xCoor, 0.0, xCoor, height);
  line(xCoor+400, 0.0, xCoor+400, height);
}

void keyPressed(){

  if (keyCode == ENTER){
    menuClose = !menuClose;
    normalVal = 1; 
  }
  
  if (key == '1' ){
    normalVal = 1;
  }
  if (key == '2'){
    normalVal = 2;
  } 
  if (key == '3'){
    normalVal = 3;
  }
  if (key == '4'){
    normalVal = 4;
  } 
  if (key == '5' ){
    normalVal = 5;
  }
  if (key == '6'){
    normalVal = 6;
  } 
  if (key == '7'){
    normalVal = 7;
  }
  if (key == '8'){
    normalVal = 8;
    xCoor = 0;
  }
  if (key == '9'){
    normalVal = 9;
    xCoor = 0;
  }
  if (key == '0'){
    normalVal = 0;
    xCoor = 0;
  }
  
  if (keyCode == LEFT){
    xCoor -=5;
    if(xCoor < 0){ xCoor = 0;}
  }
  if (keyCode == RIGHT){
    xCoor +=5;
    if(xCoor > width-400){ xCoor = width-400;}
  }
  
  /*
  1: modo normal
  2: modo grises
  3: modo umbralizado
  4: modo bordes
  */
}


//Copia unsigned byte Mat a color CVImage
void  cpMat2CVImage(Mat in_mat,CVImage out_img, int imgwidth, int imgheight){    
  byte[] data8 = new byte[imgwidth*imgheight];
  
  out_img.loadPixels();
  in_mat.get(0, 0, data8);
  
  // Cada columna
  for (int x = 0; x < imgwidth; x++) {
    // Cada fila
    for (int y = 0; y < imgheight; y++) {
      // Posición en el vector 1D
      int loc = x + y * imgwidth;
      //Conversión del valor a unsigned basado en 
      //https://stackoverflow.com/questions/4266756/can-we-make-unsigned-byte-in-java
      int val = data8[loc] & 0xFF;
      //Copia a CVImage
      out_img.pixels[loc] = color(val);
    }
  }
  out_img.updatePixels();
}
