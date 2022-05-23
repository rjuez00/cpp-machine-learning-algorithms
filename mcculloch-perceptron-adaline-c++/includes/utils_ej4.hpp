#ifndef UTILS_EJ4_H
#define UTILS_EJ4_H

#include "../includes/redneuronal.hpp"
#include "../includes/manejodatos.hpp"

void entrenamiento_perceptron(Vec2Df &atributosEntrenamiento, Vec2Df &clasesEntrenamiento, RedNeuronal &red, Capa &entradaC, Capa &salidaC, float learn_rate, float max_iters=500,  Vec2Df *atributosTest=nullptr, Vec2Df *clasesTest=nullptr);

void entrenamiento_adaline(Vec2Df &atributosEntrenamiento, Vec2Df &clasesEntrenamiento, RedNeuronal &red, Capa &entradaC, Capa &salidaC, float threshold, float learn_rate, float max_iters=500,  Vec2Df *atributosTest=nullptr, Vec2Df *clasesTest=nullptr);

void imprimir_frontera(Capa &entradaC, Capa &salidaC);

//esta función requiere una Red en la que la capa de inputs es la primera almacenada y la de outputs es la última almacenada
Vec2Df predecir(RedNeuronal &red, Vec2Df &atributosTest);

Vec2Df predecir_arg_max(RedNeuronal &red, Vec2Df &atributosTest);

float accuracy(Vec2Df &clasesEsperadas, Vec2Df &clasesPredichas);

#endif /* UTILS_EJ4_H */