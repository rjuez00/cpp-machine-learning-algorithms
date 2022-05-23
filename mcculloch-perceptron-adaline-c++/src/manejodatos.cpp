#include "../includes/manejodatos.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>


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
