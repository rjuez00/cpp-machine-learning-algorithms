#include "../includes/redneuronal.hpp"


RedNeuronal::RedNeuronal() {}
RedNeuronal::~RedNeuronal() {}

void RedNeuronal::Inicializar() {
    for (auto & capa: this->capas) {
        capa->Inicializar();
    }
}

void RedNeuronal::Anadir(Capa *capa) {
    capas.push_back(capa);
}

void RedNeuronal::Disparar() {
    for (auto & capa: this->capas) {
        capa->Disparar();
    }
}

void RedNeuronal::Propagar() {
    for (auto & capa: this->capas) {
        capa->Propagar();
    }
}
