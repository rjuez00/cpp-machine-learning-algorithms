#ifndef NEURONA_H
#define NEURONA_H

#include <vector>
#include "conexion.hpp"

#define NO_UMBRAL -999


class Neurona {
public:
    enum {
        Directa,
        Sesgo,
        McCulloch,
        Perceptron,
        Adaline
    };

    Neurona(float umbral, int tipo);
    virtual ~Neurona();

    void Inicializar(float x);
    void Conectar(Neurona *neurona, float peso);
    void Disparar();
    void Propagar();

public:
    int tipo;
    float umbral;
    float valor_entrada;
    float valor_salida;
    std::vector<Conexion> conexiones;
};

#endif /* NEURONA_H */
