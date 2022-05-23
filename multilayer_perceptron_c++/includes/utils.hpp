#ifndef UTILS_EJ4_H
#define UTILS_EJ4_H

#include "../includes/redneuronal.hpp"
#include "../includes/manejodatos.hpp"

void entrenamiento_perceptron_multicapa(Vec2Df &atributosEntrenamiento, Vec2Df &clasesEntrenamiento, RedNeuronal &red, float learn_rate, int paciencia,float max_iters=500, Vec2Df *atributosTest = nullptr, Vec2Df *clasesTest = nullptr);
float activacion_sigmoide_bipolar(float x);

void imprimir_frontera(Capa &entradaC, Capa &salidaC);

void imprimirMatrizConfusion(Vec2Df predicciones, Vec2Df clasesReales);

//esta función requiere una Red en la que la capa de inputs es la primera almacenada y la de outputs es la última almacenada
Vec2Df predecir(RedNeuronal &red, Vec2Df &atributosTest);

Vec2Df predecir_arg_max(RedNeuronal &red, Vec2Df &atributosTest);

float accuracy(Vec2Df &clasesEsperadas, Vec2Df &clasesPredichas);

#endif /* UTILS_EJ4_H */