#include "../includes/manejodatos.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>


std::tuple<Vec2Df, Vec2Df> leerAux(std::string fichero_datos) {

    int cantidadAtributos, cantidadClases;
    std::ifstream input;
    std::string line;

    input.open(fichero_datos);
    if (!input) return std::make_tuple(Vec2Df(), Vec2Df()); // error so we return empty vectors


    //leer distribucion de atributos y clases
    std::getline(input, line);
    std::istringstream iss(line);
    iss >> cantidadAtributos >> cantidadClases;

    
    std::vector<float> tempVector;
    Vec2Df atributos, clases;

    float tempRead;
    while (std::getline(input, line)) {
        std::istringstream iss(line);
        
        for (int i = 0; i<(cantidadAtributos); i++){
            iss >> tempRead;
            tempVector.push_back(tempRead);
        }
        atributos.push_back(tempVector);
        tempVector.clear();

        for (int i = 0; i<(cantidadClases); i++){
            iss >> tempRead;
            if (tempRead == 0)  tempRead = -1;
            
            tempVector.push_back(tempRead);
        }
        clases.push_back(tempVector);
        tempVector.clear();
    }
    return std::make_tuple(atributos, clases);
}


std::tuple<Vec2Df, Vec2Df, Vec2Df, Vec2Df> leer1(std::string fichero_datos, float por) {
    auto tuplaGeneral = leerAux(fichero_datos);
    
    Vec2Df atributos = std::get<0>(tuplaGeneral);
    Vec2Df clases = std::get<1>(tuplaGeneral);

    std::vector<int> indices;
    for (long unsigned int i = 0; i < atributos.size(); ++i) {
        indices.push_back(i);
    }
    std::random_shuffle(indices.begin(), indices.end());


    int n_datos_train = atributos.size()*por;

    Vec2Df atributos_entrenamiento, clases_entrenamiento, atributos_test, clases_test;

    for (auto it = std::begin(indices); it != std::begin(indices) + n_datos_train && it != std::end(indices); ++it) {
        atributos_entrenamiento.push_back(atributos[*it]);
        clases_entrenamiento.push_back(clases[*it]);
    }

    for (auto it = std::begin(indices) + n_datos_train; it != std::end(indices); ++it) {
        atributos_test.push_back(atributos[*it]);
        clases_test.push_back(clases[*it]);
    }

    return std::make_tuple(atributos_entrenamiento, clases_entrenamiento, atributos_test, clases_test);
}


std::tuple<Vec2Df, Vec2Df> leer2(std::string fichero_datos) {
    return leerAux(fichero_datos);
}


std::tuple<Vec2Df, Vec2Df, Vec2Df, Vec2Df> leer3(std::string fichero_entrenamiento, std::string fichero_test) {
    auto entrenamiento = leerAux(fichero_entrenamiento);
    auto test = leerAux(fichero_test);

    return std::tuple_cat(entrenamiento, std::move(test));
}


std::tuple<Vec2Df, Vec2Df, Vec2Df, Vec2Df, Vec2Df, Vec2Df> leer1_val(std::string fichero_datos, float proporcionValidacion, float proporcionTest) {
    if (proporcionTest + proporcionValidacion >= 1){
        return std::make_tuple(Vec2Df(), Vec2Df(), Vec2Df(), Vec2Df(), Vec2Df(), Vec2Df());
    }
    
    auto tuplaGeneral = leerAux(fichero_datos);
    
    Vec2Df atributos = std::get<0>(tuplaGeneral);
    Vec2Df clases = std::get<1>(tuplaGeneral);

    std::vector<int> indices;
    for (long unsigned int i = 0; i < atributos.size(); ++i) {
        indices.push_back(i);
    }
    std::random_shuffle(indices.begin(), indices.end());


    int n_datos_validacion = atributos.size() * proporcionValidacion;
    int n_datos_test = atributos.size() * proporcionTest;
    int n_datos_train = atributos.size() - n_datos_validacion - n_datos_test;

    Vec2Df atributos_entrenamiento, clases_entrenamiento, atributos_validacion, clases_validacion, atributos_test, clases_test;

    // Train
    for (auto it = std::begin(indices); it != std::begin(indices) + n_datos_train && it != std::end(indices); ++it) {
        atributos_entrenamiento.push_back(atributos[*it]);
        clases_entrenamiento.push_back(clases[*it]);
    }

    // Validacion
    for (auto it = std::begin(indices) + n_datos_train;  it != std::begin(indices) + n_datos_train + n_datos_validacion && it != std::end(indices); ++it) {
        atributos_validacion.push_back(atributos[*it]);
        clases_validacion.push_back(clases[*it]);
    }

    // Test
    for (auto it = std::begin(indices) + n_datos_train + n_datos_validacion;  it != std::end(indices); ++it) {
        atributos_test.push_back(atributos[*it]);
        clases_test.push_back(clases[*it]);
    }

    return std::make_tuple(atributos_entrenamiento, clases_entrenamiento, atributos_validacion, clases_validacion, atributos_test, clases_test);
}


std::tuple<Vec2Df, std::vector<float>, std::vector<float>> normalizar_train(Vec2Df atributos_train) {
    Vec2Df atributos;

    atributos.resize(atributos_train.size());
    for (auto & row : atributos_train) {
        for (long unsigned int i=0; i<row.size(); ++i) {
            atributos[i].push_back(row[i]);
        }
    }

    // Calcular means y stdevs
    std::vector<float> medias, desviaciones;
    for (auto & atrib : atributos) {
        double sum = std::accumulate(atrib.begin(), atrib.end(), 0.0);
        double mean = sum / atrib.size();
        double sq_sum = std::inner_product(atrib.begin(), atrib.end(), atrib.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / atrib.size() - mean * mean);
        
        medias.push_back(mean);
        desviaciones.push_back(stdev);
    }
    
    return std::make_tuple(normalizar_datos(atributos_train, medias, desviaciones), medias, desviaciones);
}


Vec2Df normalizar_datos(Vec2Df atributos, std::vector<float> medias, std::vector<float> desviaciones) {
    Vec2Df atributos_normalizados;

    for (auto & row : atributos) {
        std::vector<float> temp_vect;
        for (long unsigned int i=0; i<row.size(); ++i) {
            temp_vect.push_back((row[i] - medias[i])/desviaciones[i]);
        }
        atributos_normalizados.push_back(temp_vect);
    }

    return atributos_normalizados;
}


void escribirVec2Df(Vec2Df datos, std::string targetFile){
    std::ofstream output;
    std::string line;

    output.open(targetFile);
    if (!output) return;

    //output << datos[0].size() << "\n";
    for (auto & linea: datos){
        bool first = true;
        for (auto & prediccion: linea){
            if (first == true){
                first = false;
                output << prediccion;
            }
            else output << " " << prediccion;
        }
        output << "\n";
    }
}


void imprimirVec2Df(Vec2Df datos){
    //std::cout << datos[0].size() << "\n";
    for (auto & linea: datos){
        bool first = true;
        for (auto & prediccion: linea){
            if (first == true){
                first = false;
                std::cout << prediccion;
            }
            else std::cout << " " << prediccion;
        }
        std::cout << "\n";
    }

}
