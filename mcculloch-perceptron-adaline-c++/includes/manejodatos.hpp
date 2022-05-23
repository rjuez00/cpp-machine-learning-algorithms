#ifndef MANEJODATOS_H
#define MANEJODATOS_H

#include <string>
#include <vector>
#include <tuple>

typedef std::vector<std::vector<float>> Vec2Df;


std::tuple<Vec2Df, Vec2Df, Vec2Df, Vec2Df> leer1(std::string fichero_datos, float por);
std::tuple<Vec2Df, Vec2Df> leer2(std::string fichero_datos);
std::tuple<Vec2Df, Vec2Df, Vec2Df, Vec2Df> leer3(std::string fichero_entrenamiento, std::string fichero_test);

void escribirVec2Df(Vec2Df datos, std::string targetFile);
void imprimirVec2Df(Vec2Df datos);
#endif /* MANEJODATOS_H */