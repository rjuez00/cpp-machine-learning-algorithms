#include "../includes/redneuronal.hpp"
#include "../includes/manejodatos.hpp"
#include "../includes/utils_ej4.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <random>
#include <cstdlib>



// programa input output
int main(int argc, char** argv) {  
    srand(time(NULL));
    
    //COMMAND LINE CONTROL
    if (argc < 4){
        //./logicos inputfile learnrate umbral
        std::cout << "Haz make ayuda_perceptron para entender el uso de argumentos\n";
        std::cout << "ejemplo " << argv[0] << "inputfile(string) learn_rate(float) umbral(float)\n";
    }

    float learn_rate = atof(argv[2]);
    float umbral = atof(argv[3]);

    //LECTURA ARCHIVOS
    Vec2Df atributosEntrenamiento, clasesEntrenamiento;
    std::tie(atributosEntrenamiento, clasesEntrenamiento) = leer2(argv[1]);

    if (atributosEntrenamiento.size() == 0 || clasesEntrenamiento.size() == 0) {
        std::cout << "Error leyendo el archivo de entrada\n";
        exit(1);
    }
    float n_atribs = atributosEntrenamiento[0].size();
    float n_clases = clasesEntrenamiento[0].size();

    //INCIALIZACIÓN DE RED
    RedNeuronal red;
    Capa entradaC, salidaC;
    std::vector<Neurona*> neuronas;


    for (int i=0; i<n_atribs; ++i) {
        Neurona *neur = new Neurona(NO_UMBRAL, Neurona::Directa);
        neuronas.push_back(neur);
        entradaC.Anadir(neur);
    }
    Neurona *sesgo = new Neurona(NO_UMBRAL, Neurona::Sesgo);
    neuronas.push_back(sesgo);
    entradaC.Anadir(sesgo);
    
    for (int i=0; i<n_clases; ++i) {
        Neurona *neur = new Neurona(umbral, Neurona::Perceptron);
        neuronas.push_back(neur);
        salidaC.Anadir(neur);
    }

    entradaC.Conectar(&salidaC, 0.0, 0.0);

    red.Anadir(&entradaC);
    red.Anadir(&salidaC);

    //EJECUCION
    entrenamiento_perceptron(atributosEntrenamiento, clasesEntrenamiento, red, entradaC, salidaC, learn_rate);
    
    //ACCURACY
    Vec2Df prediccionTemp = predecir_arg_max(red, atributosEntrenamiento);
    std::cout << "#ACCURACY FINAL: " << accuracy(clasesEntrenamiento, prediccionTemp) << "\n";

    //IMPRIMIR RECTA DIVISION
    imprimir_frontera(entradaC, salidaC);  

    // Libera la memoria de las neuronas
    for (auto & neurona : neuronas) {
        delete neurona;
    }
    
    // Devolver pesos en terminal
    // modo 2 de lectura
}
