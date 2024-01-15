# --------------------
# Comandos
# --------------------
# o sinal de menos significa "se der erro, ignora e não para"
# ou seja, quando ele não conseguir deletar um arquivo (já
# deletado, ou não gerado), ele ignora.
RM 		:= -rm
CC 		:= g++
# --------------------
# Pastas
# --------------------
SRCDIR 	:= src
OBJDIR 	:= obj
BINDIR 	:= bin
#TSTDIR  := test
INCDIR  := inc
LIBDIR  := lib
TESTLIBDIR:=testlib
# --------------------
# Arquivos
# --------------------
SRC 	:= $(wildcard $(SRCDIR)/*.cpp)
MAIN 	:= $(SRCDIR)/main.c
MAINO 	:= $(OBJDIR)/main.o
TARGET 	:= $(BINDIR)/main
#TSTGET  := $(BINDIR)/test
# -lXXX vai procurar um arquivo com nome libXXX.a
LIB		:= $(wildcard $(LIBDIR)/*.o) $(wildcard $(LIBDIR)/*.a) `pkg-config --libs opencv4`
#TESTLIB := $(wildcard $(TESTLIBDIR)/*.o) $(wildcard $(TESTLIBDIR)/*.a)
OBJ 	:= $(SRC:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
# --------------------
# Flags para o compilador
# --------------------
# Sobre as flags utilizadas: I é para a diretiva #include encontrar arquivos em
# tal pasta. -Wall pede todos os avisos (Warning:all) e -g ajuda no debugger
# porque preserva o número da linha de código.
CFLAGS 	:= -Iinc -Wall -g `pkg-config --cflags opencv4` -g
CXXFLAGS:= -Iinc -Wall -g `pkg-config --cflags opencv4` -g
#TSTFLAG :=
LNKFLAG := -O2 -Os -static -g
# --------------------
# Variáveis rel. a Testes
# --------------------
#TSTSRC	:= $(wildcard $(TSTDIR)/*.c)
#TSTOBJ	:= $(TSTSRC:$(TSTDIR)/%.c=$(OBJDIR)/%.to)
# Quando queremos compilar o teste, não queremos duas funções main(), senão daria erros.
# Então para compilar, filtramos fora o arquivo main.o (o main.c em sua versão pré-comp)
#NMAINCOBJ:= $(filter-out $(MAINO), $(OBJ))

# --------------------
# Regras de compilação
# --------------------

# all é o default. O comando 'make' sem argumentos cai aqui.
all: $(TARGET)

# Usamos o comando 'make clean' quando mudamos o tamanho de alguma coisa e
# queremos recompilar TUDO. Por padrão, ele só recompilaria arquivos .c que mudaram
clean:
	$(RM) $(OBJ) $(TSTOBJ)

# A 'receita' para arquivos obj (que são os .o) envolve os arquivos .c correspondentes
obj/%.o: src/%.cpp
	$(CC)  $(CXXFLAGS) -c $(@:$(OBJDIR)/%.o=$(SRCDIR)/%.cpp) -o $@

# O arquivo executável final depende de todos os arquivos .o (que não são de testes)
$(TARGET) : $(OBJ)
	$(CC) -o $(TARGET) $(OBJ) $(LIB)
