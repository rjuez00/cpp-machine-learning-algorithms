#include "../includes/utils.hpp"
#include "../includes/conexion.hpp"
#include <iostream>
#include <cmath>
#include <climits>


float activacion_sigmoide_bipolar(float x) {
    return (2 / (1 + std::exp(-x))) - 1;
}


float derivada_sigmoide_bipolar(float x) {
    return 0.5 * (1 + activacion_sigmoide_bipolar(x)) * (1 - activacion_sigmoide_bipolar(x));
}


void feed_forward(RedNeuronal &red, std::vector<float> &inputs){

    Capa *entradaC = red.capas[0];

    red.Inicializar(); // TODO: Beni chequear si hace falta al final
    for (long unsigned int i=0; i<red.capas.size(); ++i) {

        // Pone los datos en las neuronas de entrada
        for(long unsigned int j=0; j < inputs.size(); j++) {
            entradaC->neuronas[j]->valor_entrada = inputs[j];
        }

        red.Disparar();
        red.Inicializar();
        red.Propagar();
    }
}


float calcular_ECM(RedNeuronal &red, Vec2Df &atributos, Vec2Df &clases){
    // ECM DE LA P1 que estaba dentro de entrenamiento
    float ecm = 0.0;
    int patrones_en_epoca = atributos.size();
    Capa *salidaC = red.capas[red.capas.size() - 1];
    for (long unsigned int i=0; i < atributos.size(); ++i) {
        
        feed_forward(red, atributos[i]);

        // Vuelve a pasar las entradas y calcula el ECM SIN modificar los pesos
        float err_cuadr_patron = 0.0;
        for (long unsigned int j=0; j < clases[i].size(); j++) {
            err_cuadr_patron += pow((clases[i][j] - salidaC->neuronas[j]->valor_salida), 2);
        }
        
        ecm += err_cuadr_patron / patrones_en_epoca;
    }
    
    return ecm;
}


void imprimirMatrizConfusion(Vec2Df predicciones, Vec2Df clasesReales) {
    if (predicciones.size() == 0 || clasesReales.size() == 0) return;
    if (predicciones.size() != clasesReales.size() || predicciones[0].size() != clasesReales[0].size()) return;

    // Reservar espacio para la matriz. La matriz es de tam_pred x tam_pred
    int tamMatriz = predicciones[0].size();
    Vec2Df confMatrix(tamMatriz, std::vector<float>(tamMatriz, 0));

    for (unsigned long int i=0; i<predicciones.size(); ++i) {

        int claseReal = 0, clasePredicha = 0;

        for (auto & clase : clasesReales[i]) {
            if (clase == 1) break;
            else ++claseReal;
        }

        for (auto & clase : predicciones[i]) {
            if (clase == 1) break;
            else ++clasePredicha;
        }

        confMatrix[claseReal][clasePredicha] += 1;

    }

    for (auto & row : confMatrix) {
        std::cout << "#";
        for (auto & elem : row) {
            std::cout << "\t" << elem;
        }
        std::cout << std::endl;
    }

}


void copia_pesos(RedNeuronal &red, std::vector<std::vector<std::vector<Conexion>>> &copia_de_pesos){
    for (long unsigned int i=0; i < red.capas.size()-1; ++i)
        for (long unsigned int j=0; j < red.capas[i]->neuronas.size(); ++j)
            copia_de_pesos[i][j] = red.capas[i]->neuronas[j]->conexiones;
}


void restaura_pesos(RedNeuronal &red, std::vector<std::vector<std::vector<Conexion>>> pesos_a_restaurar){
    for (long unsigned int i=0; i < red.capas.size()-1; ++i)
        for (long unsigned int j=0; j < red.capas[i]->neuronas.size(); ++j)
            red.capas[i]->neuronas[j]->conexiones = pesos_a_restaurar[i][j];
}


void entrenamiento_perceptron_multicapa(Vec2Df &atributosEntrenamiento, Vec2Df &clasesEntrenamiento, RedNeuronal &red, float learn_rate, int paciencia, float max_iters, Vec2Df *atributosValidacion, Vec2Df *clasesValidacion) {
    int iter = 0, lastBestAntiguedad = -1;
    Capa *salidaC = red.capas[red.capas.size() - 1];
    float lastBestAccuracy = 0;

    // Reservar el espacio necesario para hacer la copia de pesos
    std::vector<std::vector<std::vector<Conexion>>> lastBestPesos;
    lastBestPesos.resize(red.capas.size()-1);
    for (long unsigned int i=0; i < red.capas.size()-1; i++)
        lastBestPesos[i].resize(red.capas[i]->neuronas.size());
    
    
    while (iter < max_iters) {

        // EPOCA
        // Cambio de pesos
        for (long unsigned int i=0; i < atributosEntrenamiento.size(); ++i) {

            //std::cout << "X1 = " << atributosEntrenamiento[i][0] << " X2 = " << atributosEntrenamiento[i][1] << std::endl;

            // FEED FORWARD
            feed_forward(red, atributosEntrenamiento[i]);
            
            // BACKPROP
            // PASO 6
            //std::cout << "\tPASO 6:\n";
            std::vector<float> deltas, deltas_in;
            for (long unsigned int k=0; k < clasesEntrenamiento[i].size(); k++) {
                float deltaTemp = (clasesEntrenamiento[i][k] - salidaC->neuronas[k]->valor_salida) * derivada_sigmoide_bipolar(salidaC->neuronas[k]->valor_entrada);
                //std::cout << "\t\tDELTA nº" << k << " " << deltaTemp << " = (" << clasesEntrenamiento[i][k] << " - " << salidaC->neuronas[k]->valor_salida << ")*f'(" << salidaC->neuronas[k]->valor_entrada << ")" << std::endl;
                deltas.push_back(deltaTemp); 
            }

            long unsigned int capa_idx = red.capas.size() - 2; // Idx de la ultima capa oculta de la red
            Capa *tempC = red.capas[capa_idx];
            for (long unsigned int j=0; j < tempC->neuronas.size(); j++) {
                for (long unsigned int k=0; k < clasesEntrenamiento[i].size(); k++) {
                    tempC->neuronas[j]->conexiones[k].peso_anterior = tempC->neuronas[j]->conexiones[k].peso;
                    tempC->neuronas[j]->conexiones[k].peso += learn_rate * deltas[k] * tempC->neuronas[j]->valor_salida; // TODO: Asegurarse
                    //std::cout << "\t\tDELTAW" << j << " = " << learn_rate * deltas[k] * tempC->neuronas[j]->valor_salida << std::endl;
                }
            }

            // PASO 7
            //std::cout << "\tPASO 7:\n";
            // es -2 para pillar la ultima capa oculta y no la de salida (que sería -1)
            for (long unsigned int n=0; n<red.capas.size() - 2; ++n) {
                tempC = red.capas[capa_idx];
                deltas_in.clear();
                for (long unsigned int j=0; j < tempC->neuronas.size(); j++) {
                    float temp_delta_in = 0.0;
                    //std::cout << "\t\tDELTA_IN nº" << j << "= ";
                    
                    // Para iterar por las neuronas de la capa anterior y tener en cuenta que
                    // todas menos la ultima capa tiene un bias
                    long unsigned int neuronasSource = red.capas[capa_idx+1]->neuronas.size() - 1;
                    if (capa_idx == red.capas.size() -2) {
                        ++neuronasSource;
                    }
                    //neuronasSource = clasesEntrenamiento[i].size();

                    for (long unsigned int k=0; k < neuronasSource ; k++) { // TODO: AREGLAAADOOOOOO PROBAAAAAAAR
                        //std::cout << " + " << deltas[k] << " * " << tempC->neuronas[j]->conexiones[k].peso_anterior;
                        temp_delta_in += deltas[k] * tempC->neuronas[j]->conexiones[k].peso_anterior;
                    }
                    deltas_in.push_back(temp_delta_in);
                    //std::cout << " = " << temp_delta_in << std::endl;
                }
                deltas.clear();
                for (long unsigned int j=0; j < tempC->neuronas.size() - 1; j++) {
                    deltas.push_back(deltas_in[j] * derivada_sigmoide_bipolar(tempC->neuronas[j]->valor_entrada));
                    //std::cout << "\t\tDELTA nº" << j << " = " << deltas[j] << std::endl;
                }

                --capa_idx;
                tempC = red.capas[capa_idx];
                for (long unsigned int h=0; h < tempC->neuronas.size(); h++) {
                    for (long unsigned int j=0; j < red.capas[capa_idx+1]->neuronas.size() - 1; j++) {
                        tempC->neuronas[h]->conexiones[j].peso_anterior = tempC->neuronas[h]->conexiones[j].peso;
                        tempC->neuronas[h]->conexiones[j].peso += learn_rate * deltas[j] * tempC->neuronas[h]->valor_salida; // TODO: Asegurarse
                        //std::cout << "\t\tvariacion_v i("<<h<<")j("<<j<<") = " << learn_rate * deltas[j] * tempC->neuronas[h]->valor_salida << " = " <<learn_rate<< " * " <<deltas[j]<< " * "<< tempC->neuronas[h]->valor_salida << std::endl;
                    }
                }
            }
        }

        //ECM
        float ecm = calcular_ECM(red, atributosEntrenamiento, clasesEntrenamiento);

        // CALCULAR ACCURACY
        //VALIDACION
        float accuracyEpochTrain, accuracyEpochValidacion = -1;
        if (clasesValidacion != nullptr && atributosValidacion != nullptr){
            Vec2Df prediccionTemp = predecir_arg_max(red, *atributosValidacion);
            accuracyEpochValidacion = accuracy(*clasesValidacion, prediccionTemp);

            if (paciencia >= 0) {
                
                
                if (accuracyEpochValidacion > lastBestAccuracy) {
                    lastBestAntiguedad = 0;
                    lastBestAccuracy = accuracyEpochValidacion;
                    copia_pesos(red, lastBestPesos);
                } else if (lastBestAntiguedad > paciencia) {
                    restaura_pesos(red, lastBestPesos);
                    return;
                }

                ++lastBestAntiguedad;
            }
            
        }
        
        

        //TRAIN
        Vec2Df prediccionTemp = predecir_arg_max(red, atributosEntrenamiento);
        accuracyEpochTrain = accuracy(clasesEntrenamiento, prediccionTemp);

        //IMPRIMIR RESULTADOS    
        //std::cout << "#\niteration: "<< iter << " #\n\tecm:" << ecm << " #\n\taccuracyTrain:"<<accuracyEpochTrain << " #\n\taccuracyValidacion:" << accuracyEpochValidacion << std::endl;
        std::cout << iter << " "<< ecm << " "<< accuracyEpochTrain << " "<< accuracyEpochValidacion << std::endl;
        imprimirMatrizConfusion(prediccionTemp, clasesEntrenamiento);
        std::cout << "#" << std::endl;

        ++iter;
    }

}


Vec2Df predecir_arg_max(RedNeuronal &red, Vec2Df &atributosTest) {
    Capa* salidaC = red.capas[red.capas.size()-1];

    Vec2Df finalOutput;
    std::vector<float> currentOutput;

    for (long unsigned int i=0; i < atributosTest.size(); ++i) {

        feed_forward(red, atributosTest[i]);

        // GUARDAR
        if (salidaC->neuronas.size() == 1) { // Si la salida solo tiene una neurona no hay que hacer arg max
            float salida = salidaC->neuronas[0]->valor_salida;
            currentOutput.push_back(salida >= 0 ? 1 : -1);
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
