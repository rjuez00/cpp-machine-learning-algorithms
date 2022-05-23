#ifndef REDNEURONAL_H
#define REDNEURONAL_H

#include <vector>
#include "capa.hpp"


class RedNeuronal {
public:

    RedNeuronal();
    virtual ~RedNeuronal();

    void Inicializar();
    void Anadir(Capa *capa);
    void Disparar();
    void Propagar();

public:
    std::vector<Capa*> capas;
};

#endif /* REDNEURONAL_H */
