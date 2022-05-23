#include "../includes/utils_ej4.hpp"
#include <iostream>
#include <cmath>
#include <climits>


void entrenamiento_perceptron(Vec2Df &atributosEntrenamiento, Vec2Df &clasesEntrenamiento, RedNeuronal &red, Capa &entradaC, Capa &salidaC, float learn_rate, float max_iters, Vec2Df *atributosTest, Vec2Df *clasesTest) {
    bool pesos_no_han_cambiado = false;
    int patrones_en_epoca = atributosEntrenamiento.size();
    int iter = 0;

    std::cout << "#RATE: " << learn_rate << std::endl << "#\n";
    std::cout << "#epoch " << "ecm " << "accuracyTrain " << "accuracyTest\n" << "#\n";

    while (!pesos_no_han_cambiado && (iter < max_iters)) {
        pesos_no_han_cambiado = true;

        // EPOCA
        // Cambio de pesos
        for (long unsigned int i=0; i < atributosEntrenamiento.size(); ++i) {
            
            for(long unsigned int j=0; j < atributosEntrenamiento[i].size(); j++) {
                entradaC.neuronas[j]->valor_entrada = atributosEntrenamiento[i][j];
            }

            red.Disparar();
            red.Inicializar();
            red.Propagar();

            // Hacer que dispare la neurona de salida
            red.Disparar();
            red.Inicializar();

            for (long unsigned int j=0; j < clasesEntrenamiento[i].size(); j++) {

                if (salidaC.neuronas[j]->valor_salida != clasesEntrenamiento[i][j]) {
                    pesos_no_han_cambiado = false;

                    for (long unsigned int k=0; k < entradaC.neuronas.size() - 1; ++k) {
                        entradaC.neuronas[k]->conexiones[j].peso += learn_rate * clasesEntrenamiento[i][j] * atributosEntrenamiento[i][k];
                    }
                    // sesgo
                    entradaC.neuronas[entradaC.neuronas.size()-1]->conexiones[j].peso += learn_rate * clasesEntrenamiento[i][j];
                }
            }
        }

        // ECM
        // Vuelve a pasar las entradas y calcula el ECM SIN modificar los pesos
        float ecm = 0.0;
        for (long unsigned int i=0; i < atributosEntrenamiento.size(); ++i) {
            
            for(long unsigned int j=0; j < atributosEntrenamiento[i].size(); j++) {
                entradaC.neuronas[j]->valor_entrada = atributosEntrenamiento[i][j];
            }

            red.Disparar();
            red.Inicializar();
            red.Propagar();

            // Hacer que dispare la neurona de salida
            red.Disparar();
            red.Inicializar();

            float err_cuadr_patron = 0.0;
            for (long unsigned int j=0; j < clasesEntrenamiento[i].size(); j++) {
                err_cuadr_patron += pow((clasesEntrenamiento[i][j] - salidaC.neuronas[j]->valor_salida), 2);
            }
            ecm += err_cuadr_patron / patrones_en_epoca;
        }

        // CALCULAR ACCURACY
        float accuracyEpochTrain, accuracyEpochTest = -1;
        if (clasesTest != nullptr && atributosTest != nullptr){
            Vec2Df prediccionTemp = predecir_arg_max(red, *atributosTest);
            accuracyEpochTest = accuracy(*clasesTest, prediccionTemp);
        }
    
        Vec2Df prediccionTemp = predecir_arg_max(red, atributosEntrenamiento);
        accuracyEpochTrain = accuracy(clasesEntrenamiento, prediccionTemp);
    
        std::cout << iter << " " << ecm << " "<<accuracyEpochTrain << " " << accuracyEpochTest << std::endl;

        ++iter;
    }
}


void entrenamiento_adaline(Vec2Df &atributosEntrenamiento, Vec2Df &clasesEntrenamiento, RedNeuronal &red, Capa &entradaC, Capa &salidaC, float threshold, float learn_rate, float max_iters,Vec2Df *atributosTest, Vec2Df *clasesTest) {
    float max_cambio_pesos = 999999.0;
    int patrones_en_epoca = atributosEntrenamiento.size();
    int iter = 0;
    float ecm;
    std::vector<float> pesos;

    std::cout << "#RATE: " << learn_rate << " THRES: " << threshold << std::endl << "#\n";
    std::cout << "#epoch " << "ecm " << "accuracyTrain " << "accuracyTest\n" << "#\n";

    while ((max_cambio_pesos > threshold) && (iter < max_iters)) {

        // Guarda todos los pesos en un vector para compararlos con los de despues de la epoca
        for (long unsigned int s=0; s < salidaC.neuronas.size(); s++) {
            for (long unsigned int e=0; e < entradaC.neuronas.size(); ++e) {
                pesos.push_back(entradaC.neuronas[e]->conexiones[s].peso);
            }
        }

        // EPOCA
        // Cambio de pesos
        for (long unsigned int i=0; i < atributosEntrenamiento.size(); ++i) {
            
            for(long unsigned int j=0; j < atributosEntrenamiento[i].size(); j++) {
                entradaC.neuronas[j]->valor_entrada = atributosEntrenamiento[i][j];
            }

            red.Disparar();
            red.Inicializar();
            red.Propagar();

            // Hacer que dispare la neurona de salida
            red.Disparar();
            
            for (long unsigned int j=0; j < clasesEntrenamiento[i].size(); j++) {

                for (long unsigned int k=0; k < entradaC.neuronas.size() - 1; ++k) {
                    entradaC.neuronas[k]->conexiones[j].peso += learn_rate * (clasesEntrenamiento[i][j] - salidaC.neuronas[j]->valor_entrada) * atributosEntrenamiento[i][k];
                }

                // sesgo
                entradaC.neuronas[entradaC.neuronas.size() - 1]->conexiones[j].peso += learn_rate * (clasesEntrenamiento[i][j] - salidaC.neuronas[j]->valor_entrada);
            }

            red.Inicializar();
        }

        //ECM
        ecm = 0.0;
        for (long unsigned int i=0; i < atributosEntrenamiento.size(); ++i) {
            
            for(long unsigned int j=0; j < atributosEntrenamiento[i].size(); j++) {
                entradaC.neuronas[j]->valor_entrada = atributosEntrenamiento[i][j];
            }

            red.Disparar();
            red.Inicializar();
            red.Propagar();

            // Hacer que dispare la neurona de salida
            red.Disparar();

            // Vuelve a pasar las entradas y calcula el ECM SIN modificar los pesos
            float err_cuadr_patron = 0.0;
            for (long unsigned int j=0; j < clasesEntrenamiento[i].size(); j++) {
                err_cuadr_patron += pow((clasesEntrenamiento[i][j] - salidaC.neuronas[j]->valor_entrada), 2);
            }
            ecm += err_cuadr_patron / patrones_en_epoca;

            red.Inicializar();
        }

        // CALCULAR ACCURACY
        float accuracyEpochTrain, accuracyEpochTest = -1;
        if (clasesTest != nullptr && atributosTest != nullptr){
            Vec2Df prediccionTemp = predecir_arg_max(red, *atributosTest);
            accuracyEpochTest = accuracy(*clasesTest, prediccionTemp);
        }
    
        Vec2Df prediccionTemp = predecir_arg_max(red, atributosEntrenamiento);
        accuracyEpochTrain = accuracy(clasesEntrenamiento, prediccionTemp);
    
        // Vemos cuanto han cambiado los pesos desde el comienzo de una epoca y
        // obtenemos el cambio de pesos mas grande (en valor absoluto)
        max_cambio_pesos = 0.0;
        int idx = 0;
        for (long unsigned int s=0; s < salidaC.neuronas.size(); s++) {
            for (long unsigned int e=0; e < entradaC.neuronas.size(); ++e) {
                float temp_delta_peso = std::abs(pesos[idx] - entradaC.neuronas[e]->conexiones[s].peso);
                max_cambio_pesos = std::max(max_cambio_pesos, temp_delta_peso);
                ++idx;
            }
        }
        pesos.clear();

        std::cout << iter << " " << ecm << " "<< accuracyEpochTrain << " " << accuracyEpochTest << std::endl;

        ++iter;
    }
}

void imprimir_frontera(Capa &entradaC, Capa &salidaC) {
    for (long unsigned int i=0; i < salidaC.neuronas.size(); i++) {     
        std::cout << "# ";

        for(long unsigned int k = 0; k < entradaC.neuronas.size(); k++) {
            if (k == entradaC.neuronas.size() -1 ) std::cout << entradaC.neuronas[k]->conexiones[i].peso;
            else std::cout << "(X" << k << "*" << entradaC.neuronas[k]->conexiones[i].peso << ") + ";
        }
        std::cout << " = 0\n";
    }
}

Vec2Df predecir_arg_max(RedNeuronal &red, Vec2Df &atributosTest) {
    Capa* entradaC = red.capas[0];
    Capa* salidaC = red.capas[1];

    Vec2Df finalOutput;
    std::vector<float> currentOutput;

    for (long unsigned int i=0; i < atributosTest.size(); ++i) {
        // INPUT
        for(long unsigned int j=0; j < atributosTest[i].size(); j++) {
            entradaC->neuronas[j]->valor_entrada = atributosTest[i][j];  
        }
        // EJECUCION
        red.Disparar();
        red.Inicializar();
        red.Propagar();

        // GUARDAR
        if (salidaC->neuronas.size() == 1) { // Si la salida solo tiene una neurona no hay que hacer arg max

            red.Disparar(); // Hacer que dispare la neurona de salida
            red.Inicializar();
            currentOutput.push_back(salidaC->neuronas[0]->valor_salida);

        } else { // Solo la neurona con mayor y_out es 1, el resto -1 (ARGMAX)

            // Guarda los valores de entrada de las neuronas de salida
            float salida_max = INT_MIN;
            long unsigned int idx_salida_max = 0;
            for(long unsigned int i=0; i < salidaC->neuronas.size(); ++i) {
                if (salidaC->neuronas[i]->valor_entrada > salida_max) {
                    salida_max = salidaC->neuronas[i]->valor_entrada;
                    idx_salida_max = i;
                }
            }
            // Asigna un valor de 1 a la neurona con un mayor valor de entrada y -1 al resto
            for(long unsigned int i=0; i < salidaC->neuronas.size(); ++i) {
                if (i == idx_salida_max) {
                    currentOutput.push_back(1);
                } else {
                    currentOutput.push_back(-1);
                }
            }
        }
        
        finalOutput.push_back(currentOutput);
        currentOutput.clear();
    }
    return finalOutput;
}

float accuracy(Vec2Df &clasesEsperadas, Vec2Df &clasesPredichas){

    if (clasesEsperadas.size() != clasesPredichas.size()) return -1;

    float errores = 0;
    for(long unsigned int i = 0; i < clasesEsperadas.size(); i++) {
        if(clasesEsperadas[i] != clasesPredichas[i]) {
            errores++;
        } 
    }

    return (clasesEsperadas.size() - errores)/clasesEsperadas.size();
}
