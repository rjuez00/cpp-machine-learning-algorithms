#ifndef CAPA_H
#define CAPA_H

#include <vector>
#include "neurona.hpp"


class Capa {
public:
    Capa();
    virtual ~Capa();

    void Inicializar();
    void Anadir(Neurona *neurona);
    void Conectar(Capa *Capa, float peso_min, float peso_max);
    void Conectar(Neurona *neurona, float peso_min, float peso_max);
    void Disparar();
    void Propagar();

public:
    std::vector<Neurona*> neuronas;
};

#endif /* CAPA_H */