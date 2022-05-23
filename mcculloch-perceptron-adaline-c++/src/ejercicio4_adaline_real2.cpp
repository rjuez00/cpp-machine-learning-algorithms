#include "../includes/redneuronal.hpp"
#include "../includes/manejodatos.hpp"
#include "../includes/utils_ej4.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <random>




// programa input output
int main(int argc, char** argv) {
    srand(time(NULL));
    // recibe train y test files por separado
    // generar archivo con predicciones
    // modo 3 de lectura


    srand(time(NULL));
    
    //COMMAND LINE CONTROL
    if (argc < 6){
        std::cout << "Haz make ayuda_perceptron para entender el uso de argumentos\n";
        std::cout << "ejemplo " << argv[0] << "inputfile_train(string) inputfile_test(string) output_file(string) learnrate(float) threshold_adaline(float)\n";
        exit(1);
    }

    float learn_rate = atof(argv[4]);
    float threshold = atof(argv[5]);

    Vec2Df atributosEntrenamiento, clasesEntrenamiento, atributosTest, clasesTest;
    std::tie(atributosEntrenamiento, clasesEntrenamiento, atributosTest, clasesTest) = leer3(argv[1], argv[2]);
    
    if (atributosEntrenamiento.size() == 0 || clasesEntrenamiento.size() == 0) {
        std::cout << "Error leyendo el archivo de entrada\n";
        exit(1);
    }
    int n_atribs = atributosEntrenamiento[0].size();
    int n_clases = clasesEntrenamiento[0].size();

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
        Neurona *neur = new Neurona(NO_UMBRAL, Neurona::Adaline);
        neuronas.push_back(neur);
        salidaC.Anadir(neur);
    }

    entradaC.Conectar(&salidaC, -0.5, 0.5);
    //entradaC.Conectar(&salidaC, 1,1);

    red.Anadir(&entradaC);
    red.Anadir(&salidaC);
    
    //EJECUCION
    entrenamiento_adaline(atributosEntrenamiento, clasesEntrenamiento, red, entradaC, salidaC, threshold, learn_rate, 5000);

    //GENERA PREDICCIONES TEST
    Vec2Df prediccionesTest = predecir_arg_max(red, atributosTest);
    escribirVec2Df(prediccionesTest, argv[3]);
    
    //accuracy EN EL INPUT ENTRENAMIENTO
    Vec2Df prediccionesEntrenamiento = predecir_arg_max(red, atributosEntrenamiento);
    std::cout << "#accuracy entrenamiento final: " << accuracy(clasesEntrenamiento, prediccionesEntrenamiento) << "\n";

    //IMPRIMIR RECTA DIVISION
    imprimir_frontera(entradaC, salidaC);

    // Libera la memoria de las neuronas
    for (auto & neurona : neuronas) {
        delete neurona;
    }

}
