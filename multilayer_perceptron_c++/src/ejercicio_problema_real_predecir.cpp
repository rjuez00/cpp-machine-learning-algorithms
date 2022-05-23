#include "../includes/redneuronal.hpp"
#include "../includes/manejodatos.hpp"
#include "../includes/utils.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <random>
#include <cstdlib>
#include <cmath>


std::vector<int> getDefinicionCapasOcultas(std::string argumentoCapasOcultas){
    std::stringstream test(argumentoCapasOcultas);
    std::string segment;
    std::vector<std::string> seglist;

    while(std::getline(test, segment, ',')) seglist.push_back(segment);

    std::vector<int> definicionCapasOcultas;
    for (auto & tamanoCapaOculta: seglist) definicionCapasOcultas.push_back(atoi(tamanoCapaOculta.c_str()));

    return definicionCapasOcultas;
}



// programa input output
int main(int argc, char** argv) {  
    srand(time(NULL));
    
    //COMMAND LINE CONTROL
    if (argc < 9){
        std::cout << argv[0] << "inputfileTrain inputfileTest outputFile learnrate capasOcultas n_iters paciencia normalizar_o_no(1 o -1)\n";
        std::cout << "ejemplo " << argv[0] << "TODO\n";
        exit(1);
    }

    float learn_rate = atof(argv[4]);
    std::vector<int> definicionCapasOcultas = getDefinicionCapasOcultas(argv[5]);


    int n_iters = atoi(argv[6]);
    int paciencia = atoi(argv[7]);
    int normalizarTemp = atoi(argv[8]);
    bool normalizar = false;
    if (normalizarTemp == 1) normalizar = true;
    else if (normalizarTemp == -1) normalizar = false;
    else {
        std::cout << "El argumento de normalizar solo puede ser 1 o -1";
        exit(1);
    }
    


    
    //LECTURA ARCHIVOS
    Vec2Df atributosEntrenamiento, clasesEntrenamiento, atributosTest, clasesTest;
    std::tie(atributosEntrenamiento, clasesEntrenamiento, atributosTest, clasesTest) = leer3(argv[1], argv[2]);


    if (atributosEntrenamiento.size() == 0 || clasesEntrenamiento.size() == 0) {
        std::cout << "Error leyendo el archivo de entrada\n";
        exit(1);
    }


    
    //NORMALIZACIÓN OPCIONAL
    if (normalizar == true){
        std::vector<float> mediaNorm, desviacionNorm;
        std::tie(atributosEntrenamiento, mediaNorm, desviacionNorm) = normalizar_train(atributosEntrenamiento);
        atributosTest = normalizar_datos(atributosTest, mediaNorm, desviacionNorm);
    }

    
    int n_atribs = atributosEntrenamiento[0].size();
    int n_clases = clasesEntrenamiento[0].size();

    //INCIALIZACIÓN DE RED
    RedNeuronal red;
    int idx_capa = 0;

    // Capa entrada
    red.Anadir(new Capa());
    for (int i=0; i<n_atribs; ++i) {
        red.capas[idx_capa]->Anadir(new Neurona(NO_UMBRAL, Neurona::Directa));
    }
    ++idx_capa;

    // Capas ocultas
    for (auto & neurs_en_ocult : definicionCapasOcultas) {
        red.Anadir(new Capa());
        for (int i=0; i<neurs_en_ocult; ++i) {
            red.capas[idx_capa]->Anadir(new Neurona(NO_UMBRAL, Neurona::SigmoideBipolar));
        }
        ++idx_capa;
    }

    // Anadir sesgos y conectarlos a la siguiente capa
    // No se anaden antes para evitar que se creen
    // conexiones de una capa anterior al sesgo de la siguiente
    // TODO: Mover comment arriba

    // Capa Salida
    red.Anadir(new Capa());
    for (int i=0; i<n_clases; ++i) {
        red.capas[idx_capa]->Anadir(new Neurona(NO_UMBRAL, Neurona::SigmoideBipolar));
    }

    // Conectar capas
    for (long unsigned int i=0; i<1 + definicionCapasOcultas.size(); ++i) {
        red.capas[i]->Anadir(new Neurona(NO_UMBRAL, Neurona::Sesgo));
        red.capas[i]->Conectar(red.capas[i+1], -0.5, 0.5);
    }

    //EJECUCION
    entrenamiento_perceptron_multicapa(atributosEntrenamiento, clasesEntrenamiento, red, learn_rate, paciencia, n_iters);

        
    //GENERA PREDICCIONES TEST
    Vec2Df prediccionesTest = predecir_arg_max(red, atributosTest);
    escribirVec2Df(prediccionesTest, argv[3]);
    
    //accuracy EN EL INPUT ENTRENAMIENTO
    Vec2Df prediccionesEntrenamiento = predecir_arg_max(red, atributosEntrenamiento);
    std::cout << "#normalizar está a " << normalizar << std::endl;
    std::cout << "#accuracy entrenamiento final: " << accuracy(clasesEntrenamiento, prediccionesEntrenamiento) << "\n";
    std::cout << "#matriz de confusion del Entrenamiento final:" << std::endl;
    imprimirMatrizConfusion(prediccionesEntrenamiento, clasesEntrenamiento);
}
