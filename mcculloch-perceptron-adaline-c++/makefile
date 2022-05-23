### MACROS

#colors
ENDC=\e[0m
MAIN_C=\e[32;1;4m
NAME_C=\e[36;1;4m
INPUT_C=\e[33;1m
OUTPUT_C=\e[95;1m

#compilation
CC=g++
INCLUDE = includes
SRC = src
CXXFLAGS = -Wall -Wextra -O3 -pg
objects_libreria = neurona.o conexion.o capa.o redneuronal.o manejodatos.o utils_ej4.o

all: all_perceptron_logicos clean


### COMUN
$(objects_libreria): %.o: $(SRC)/%.cpp $(INCLUDE)/%.hpp
	@echo "#$(MAIN_C)COMPILANDO$(NAME_C) $< $(ENDC)"
	@$(CC) -c $(CXXFLAGS) $< -o $@

clean:
	@echo "#$(MAIN_C)LIMPIANDO! $(ENDC)"
	@rm -rf *.o *.bin *.output gmon.out

numero_pareja:
	@echo "#el numero de pareja es 7"


### EJERCICIO 2
ejercicio2_executable=ejercicio2.bin
ejercicio2_inputfile="ej2files/ejercicio2.input"
ejercicio2_outputfile="ejercicio2.output"

ayuda_mp:
	@echo "./$(ejercicio2_executable) $(INPUT_C)input_filename$(ENDC)(string) $(OUTPUT_C)output_filename$(ENDC)(string)"
	@echo "ARGUMENTOS:"
	@echo "    $(INPUT_C)input_filename$(ENDC): Fichero de entrada donde figuran las entradas de la red"
	@echo "    $(OUTPUT_C)output_filename$(ENDC): Fichero donde se guarda el output de la red. Si no se especifica, se imprime en el terminal"
	@echo "    		$(OUTPUT_C)nota$(ENDC): Si ejecutas 'ejecuta_mp' se guarda el archivo en ejecuta2.output, si no especificas archivo de output al llamar al ejercicio2.bin se muestra por terminal"

compila_mp: $(objects_libreria) $(SRC)/ejercicio2.cpp
	@echo "#$(MAIN_C)COMPILANDO EJECUTABLE! $(ENDC)"
	@$(CC) -c $(CXXFLAGS) $(SRC)/ejercicio2.cpp -o ejercicio2.o
	@$(CC) $(CXXFLAGS) -o $(ejercicio2_executable) ejercicio2.o $(objects_libreria)

ejecuta_mp:
	@echo "#$(MAIN_C)EJECUTANDO$(NAME_C) $(ejercicio2_executable) $(ENDC)"
	@./$(ejercicio2_executable) $(ejercicio2_inputfile) $(ejercicio2_outputfile)

all_mp: compila_mp ejecuta_mp



#EJERCICIO 4 PROBLEMA REAL 1 COMANDOS
compilar: compila_perceptron_logicos compila_perceptron_real1 compila_perceptron_real2 compila_adaline_logicos compila_adaline_real1 compila_adaline_real2

ayuda_perceptron:
	@echo  "./ejercicio4_perceptron_real1.bin $(INPUT_C)inputfile(string)$(ENDC) $(OUTPUT_C)learn_rate(float)$(ENDC) $(OUTPUT_C)umbral(float)$(ENDC) $(OUTPUT_C)proporcionTrain(float)$(ENDC)"
	@echo "ARGUMENTOS:"
	@echo "    $(INPUT_C)inputfile$(ENDC): Fichero de entrada donde figuran las entradas y salidas de la red"
	@echo "    $(OUTPUT_C)learn_rate$(ENDC): Tasa de aprendizaje de las neuronas de la red"
	@echo "    $(OUTPUT_C)umbral$(ENDC): Umbral de las neuronas de salida"
	@echo "    $(OUTPUT_C)proporcionTrain$(ENDC): Proporción de los datos del inputfile que se usan para el entrenamiento"


ayuda_adaline:
	@echo  "./ejercicio4_adaline_real1.bin $(INPUT_C)inputfile(string)$(ENDC) $(OUTPUT_C)learn_rate(float)$(ENDC) $(OUTPUT_C)proporcionTrain(float)$(ENDC) $(OUTPUT_C)threshold_adaline(float)$(ENDC)"
	@echo "ARGUMENTOS:"
	@echo "    $(INPUT_C)inputfile$(ENDC): Fichero de entrada donde figuran las entradas y salidas de la red"
	@echo "    $(OUTPUT_C)learn_rate$(ENDC): Tasa de aprendizaje de las neuronas de la red"
	@echo "    $(OUTPUT_C)proporcionTrain$(ENDC): Proporción de los datos del inputfile que se usan para el entrenamiento"
	@echo "    $(OUTPUT_C)threshold_adaline$(ENDC): Tolerancia del maximo cambio de pesos permitido para acabar de entrenar"

ejecuta_perceptron: ejecuta_perceptron_real1

ejecuta_adaline: ejecuta_adaline_real1


### EJERCICIO 4 MAINS
### PERCEPTRON
## PERCEPTRON LOGICOS
all_perceptron_logicos: compila_perceptron_logicos ejecuta_perceptron_logicos

compila_perceptron_logicos: $(objects_libreria) $(SRC)/ejercicio4_perceptron_logicos.cpp
	@echo "#$(MAIN_C)COMPILANDO EJECUTABLE! $(ENDC)"
	@$(CC) -c $(CXXFLAGS) $(SRC)/ejercicio4_perceptron_logicos.cpp -o ejercicio4_perceptron_logicos.o
	@$(CC) $(CXXFLAGS) -o ejercicio4_perceptron_logicos.bin ejercicio4_perceptron_logicos.o $(objects_libreria)

ejecuta_perceptron_logicos:
	@echo "#$(MAIN_C)EJECUTANDO$(NAME_C) ejercicio4_perceptron_logicos.bin $(ENDC)"
	@./ejercicio4_perceptron_logicos.bin "ej4files/xor.txt" 0.1 0.2



## PERCEPTRON PROBLEMA REAL 1
all_perceptron_real1: compila_perceptron_real1 ejecuta_perceptron_real1

compila_perceptron_real1: $(objects_libreria) $(SRC)/ejercicio4_perceptron_real1.cpp
	@echo "#$(MAIN_C)COMPILANDO EJECUTABLE! $(ENDC)"
	@$(CC) -c $(CXXFLAGS) $(SRC)/ejercicio4_perceptron_real1.cpp -o ejercicio4_perceptron_real1.o
	@$(CC) $(CXXFLAGS) -o ejercicio4_perceptron_real1.bin ejercicio4_perceptron_real1.o $(objects_libreria)

ejecuta_perceptron_real1:
	@echo "#$(MAIN_C)EJECUTANDO$(NAME_C) ejercicio4_perceptron_real1.bin $(ENDC)"
	@./ejercicio4_perceptron_real1.bin "ej4files/problema_real1.txt" 0.1 0.8 0.7



## PERCEPTRON PROBLEMA REAL 2
all_perceptron_real2: compila_perceptron_real2 ejecuta_perceptron_real2

compila_perceptron_real2: $(objects_libreria) $(SRC)/ejercicio4_perceptron_real2.cpp
	@echo "#$(MAIN_C)COMPILANDO EJECUTABLE! $(ENDC)"
	@$(CC) -c $(CXXFLAGS) $(SRC)/ejercicio4_perceptron_real2.cpp -o ejercicio4_perceptron_real2.o
	@$(CC) $(CXXFLAGS) -o ejercicio4_perceptron_real2.bin ejercicio4_perceptron_real2.o $(objects_libreria)

ejecuta_perceptron_real2:
	@echo "#$(MAIN_C)EJECUTANDO$(NAME_C) ejercicio4_perceptron_real2.bin $(ENDC)"
	@./ejercicio4_perceptron_real2.bin "ej4files/problema_real2.txt" "ej4files/problema_real2_no_etiquetados.txt" "predicciones/prediccion_perceptron.txt" 0.001 0.4




### ADALINE
## ADALINE LOGICOS
all_adaline_logicos: compila_adaline_logicos ejecuta_adaline_logicos

compila_adaline_logicos: $(objects_libreria) $(SRC)/ejercicio4_adaline_logicos.cpp
	@echo "#$(MAIN_C)COMPILANDO EJECUTABLE! $(ENDC)"
	@$(CC) -c $(CXXFLAGS) $(SRC)/ejercicio4_adaline_logicos.cpp -o ejercicio4_adaline_logicos.o
	@$(CC) $(CXXFLAGS) -o ejercicio4_adaline_logicos.bin ejercicio4_adaline_logicos.o $(objects_libreria)

ejecuta_adaline_logicos:
	@echo "#$(MAIN_C)EJECUTANDO$(NAME_C) ejercicio4_adaline_logicos.bin $(ENDC)"
	@./ejercicio4_adaline_logicos.bin "ej4files/and.txt" 0.1 0.01



## ADALINE PROBLEMA REAL 1
all_adaline_real1: compila_adaline_real1 ejecuta_adaline_real1

compila_adaline_real1: $(objects_libreria) $(SRC)/ejercicio4_adaline_real1.cpp
	@echo "#$(MAIN_C)COMPILANDO EJECUTABLE! $(ENDC)"
	@$(CC) -c $(CXXFLAGS) $(SRC)/ejercicio4_adaline_real1.cpp -o ejercicio4_adaline_real1.o
	@$(CC) $(CXXFLAGS) -o ejercicio4_adaline_real1.bin ejercicio4_adaline_real1.o $(objects_libreria)

ejecuta_adaline_real1:
	@echo "#$(MAIN_C)EJECUTANDO$(NAME_C) ejercicio4_adaline_real1.bin $(ENDC)"
	@./ejercicio4_adaline_real1.bin "ej4files/problema_real1.txt" 0.1 0.7 0.0001



## ADALINE PROBLEMA REAL 2
all_adaline_real2: compila_adaline_real2 ejecuta_adaline_real2

compila_adaline_real2: $(objects_libreria) $(SRC)/ejercicio4_adaline_real2.cpp
	@echo "#$(MAIN_C)COMPILANDO EJECUTABLE! $(ENDC)"
	@$(CC) -c $(CXXFLAGS) $(SRC)/ejercicio4_adaline_real2.cpp -o ejercicio4_adaline_real2.o
	@$(CC) $(CXXFLAGS) -o ejercicio4_adaline_real2.bin ejercicio4_adaline_real2.o $(objects_libreria)

ejecuta_adaline_real2:
	@echo "#$(MAIN_C)EJECUTANDO$(NAME_C) ejercicio4_adaline_real2.bin $(ENDC)"
	@./ejercicio4_adaline_real2.bin "ej4files/problema_real2.txt" "ej4files/problema_real2_no_etiquetados.txt" "predicciones/prediccion_adaline.txt" 0.0001 0.0001

