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
CXXFLAGS = -Wall -Wextra -O3 # -g
objects_libreria = neurona.o conexion.o capa.o redneuronal.o manejodatos.o utils.o

all: all_problema_real_split clean


### COMUN
$(objects_libreria): %.o: $(SRC)/%.cpp $(INCLUDE)/%.hpp
	@echo "#$(MAIN_C)COMPILANDO$(NAME_C) $< $(ENDC)"
	@$(CC) -c $(CXXFLAGS) $< -o $@

clean:
	@echo "#$(MAIN_C)LIMPIANDO! $(ENDC)"
	@rm -rf *.o *.bin *.output gmon.out

numero_pareja:
	@echo "#el numero de pareja es 7"



### PROBLEMAS REALES
compila_real_split: $(objects_libreria) $(SRC)/ejercicio_problema_real_split.cpp
	@echo "#$(MAIN_C)COMPILANDO EJECUTABLE! $(ENDC)"
	@$(CC) -c $(CXXFLAGS) $(SRC)/ejercicio_problema_real_split.cpp -o ejercicio_problema_real_split.o
	@$(CC) $(CXXFLAGS) -o ejercicio_problema_real_split.bin ejercicio_problema_real_split.o $(objects_libreria)

ejecuta_real_split: compila_real_split
	@echo "#$(MAIN_C)EJECUTANDO$(NAME_C) ejercicio_problema_real_split.bin $(ENDC)"
	@./ejercicio_problema_real_split.bin data/problema_real2.txt 0.1 2 0.7 0.3 5000 10 -1

### NORMALIZAR (de momento está sin validación, ya vemos el 6 o el 4...)
ejecuta_real_normalizado: compila_real_split
	@echo "#$(MAIN_C)EJECUTANDO$(NAME_C) ejercicio_problema_real_split.bin $(ENDC)"
	@./ejercicio_problema_real_split.bin data/problema_real6.txt 0.1 2 0.6 0.2 5000 50 1


### VALIDACION
ejecuta_real_validacion: compila_real_split
	@echo "#$(MAIN_C)EJECUTANDO$(NAME_C) ejercicio_problema_real_split.bin $(ENDC)"
	@./ejercicio_problema_real_split.bin data/problema_real2.txt 0.1 2 0.6 0.2 5000 10 -1



### PROBLEMA PREDECIR
compila_real_predecir: $(objects_libreria) $(SRC)/ejercicio_problema_real_predecir.cpp
	@echo "#$(MAIN_C)COMPILANDO EJECUTABLE! $(ENDC)"
	@$(CC) -c $(CXXFLAGS) $(SRC)/ejercicio_problema_real_predecir.cpp -o ejercicio_problema_real_predecir.o
	@$(CC) $(CXXFLAGS) -o ejercicio_problema_real_predecir.bin ejercicio_problema_real_predecir.o $(objects_libreria)



# TARGETS MEMORIA
ejecuta_problema_real6: compila_real_split
	@echo "#$(MAIN_C)EJECUTANDO$(NAME_C) ejercicio_problema_real_split.bin $(ENDC)"
	@./ejercicio_problema_real_split.bin data/problema_real6.txt 0.1 20 0.7 0.3 5000 -1 1

compilar: compila_real_predecir compila_real_split
	
ayuda_retro:
	@echo "#$(MAIN_C)HAY DOS PROGRAMAS:$(ENDC)"
	@echo "#$(MAIN_C)1)$(NAME_C) ejercicio_problema_real_split.bin $(ENDC)"
	@echo "./ejercicio_problema_real_split.bin inputfile(str) learnrate(float) capasOcultas(int,int,int...) proporcionTrain(float) proporcionTest(float) n_iters(int) paciencia(int) normalizar_o_no(1 o -1)";
	@echo "\tproporcionTrain & proporcionTest: propocion de datos para el train y respectivamente, si no suman 1 entonces se usa el resto como proporcion de validación"
	@echo "\t\tejemplo de 70% train y 30% test sería poner: 0.7 0.3"
	@echo "\t\tejemplo de 70% train y 20% test y 10% validacion sería poner: 0.7 0.2"

	@echo "\n"
	@echo "#$(MAIN_C)2)$(NAME_C) ejercicio_problema_real_predecir.bin $(ENDC)"
	@echo "./ejercicio_problema_real_predecir.bin inputfileTrain(str) inputfileTest(str) outputFile(str) learnrate(float) capasOcultas(int,int,int...) n_iters(int) paciencia(int) normalizar_o_no(1 o -1)";
	
	@echo "\n"
	@echo "#$(MAIN_C)INFORMACION COMUN$(ENDC)"
	@echo "\tcapasOcultas: se permiten un número ilimitado de capas ocultas: para una sola capa oculta solo hay que poner un entero, y si queremos mas poner varios enteros con comas y sin espacios"
	@echo "\t\tejemplo de 20 neuronas y 1 capa oculta: ./ejercicio_problema_real_split.bin data/problema_real2.txt 0.1 20 0.7 0.15 5000 100 -1"
	@echo "\t\tejemplo de 2 capa ocultas de 10 y 20 neuronas: ./ejercicio_problema_real_split.bin data/problema_real2.txt 0.1 10,20 0.7 0.15 5000 100 -1"
	@echo "\n"

ejecuta_problema_real2:
	@./ejercicio_problema_real_split.bin data/problema_real2.txt 0.1 20 0.7 0.15 5000 100 -1

ejecuta_prediccion_real_2: compila_real_predecir
	@echo "#$(MAIN_C)EJECUTANDO$(NAME_C) ejercicio_problema_real_predecir.bin $(ENDC)"
	@./ejercicio_problema_real_predecir.bin data/problema_real2.txt data/problema_real2_no_etiquetados.txt prediccion_problema_real2.txt 0.1 20 300 -1 -1