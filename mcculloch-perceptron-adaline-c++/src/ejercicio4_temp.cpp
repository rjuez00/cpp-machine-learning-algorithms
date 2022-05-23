#include "../includes/redneuronal.hpp"
#include "../includes/manejodatos.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <random>

// programa input output
int main(int argc, char** argv) {
    //argumentos: TODO METER POR COMMAND LINE
    int modoLectura = 1;
    float proporcion = 0.2;
    std::string entrenamientoFile = "ej4files/and.txt"; //SI SOLO SE USA UN ARCHIVO ESTE ES EL PRINCIPAL
    std::string testFile = "ej4files/and.txt";
    //end argumentos command line

    srand(time(NULL));

    if(argc < 2){ //TODO meter en el if dependiendo del modo lectura
        std::cout << "Haga 'make ayuda_perceptron'\n";
        exit(1);
    }

    Vec2Df atributosEntrenamiento, clasesEntrenamiento, atributosTest, clasesTest;
    if(modoLectura == 1){
        if(argc < 2){ 
            std::cout << "Haga 'make ayuda_perceptron'\n";
            exit(1);
        }

        std::tie(atributosEntrenamiento, clasesEntrenamiento, atributosTest, clasesTest) = leer1(entrenamientoFile, proporcion);
    }
    else if (modoLectura == 2){
        if(argc < 2){ 
            std::cout << "Haga 'make ayuda_perceptron'\n";
            exit(1);
        }


        std::tie(atributosEntrenamiento, clasesEntrenamiento) = leer2(entrenamientoFile);
        atributosTest = atributosEntrenamiento;
        clasesTest = clasesEntrenamiento;   
    }
    else if (modoLectura == 3){
        if(argc < 2){ 
            std::cout << "Haga 'make ayuda_perceptron'\n";
            exit(1);
        }
        
        std::tie(atributosEntrenamiento, clasesEntrenamiento, atributosTest, clasesTest) = leer3(entrenamientoFile, testFile);
    }
    else {
        std::cout << "Modo de Lectura incorrecto";
        exit(1);
    }
    
    int n_atribs = 2, n_clases = 1;
    float umbral = 0.2;
    RedNeuronal red;
    Capa entradaC, salidaC;
    std::vector<Neurona*> neuronas;

    neuronas.push_back(new Neurona(NO_UMBRAL, Neurona::Sesgo));
    entradaC.Anadir(neuronas[0]);
    for (int i=0; i<n_atribs; ++i) {
        Neurona *neur = new Neurona(NO_UMBRAL, Neurona::Directa);
        entradaC.Anadir(neur);
    }
    for (int i=0; i<n_clases; ++i) {
        Neurona *neur = new Neurona(umbral, Neurona::Perceptron);
        neuronas.push_back(neur);
        salidaC.Anadir(neur);
    }

    // TODO: Ver que min y max probar
    entradaC.Conectar(&salidaC, -1.0, 1.0);

    red.Anadir(&entradaC);
    red.Anadir(&salidaC);
    
    // Libera la memoria de las neuronas
    for (auto & neurona : neuronas) {
        delete neurona;
    }
}
