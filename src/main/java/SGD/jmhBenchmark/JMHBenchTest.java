package SGD.jmhBenchmark;


import SGD.MovieRecommenderSGD;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.HashSet;
import java.util.concurrent.TimeUnit;



// É importante que a classe MovieRecommenderSGD esteja no mesmo ClassLoader
// ou que seus métodos estáticos e classes internas sejam acessíveis.

@State(Scope.Benchmark)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 10, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(value = 1, jvmArgsAppend = {"-Xms2g", "-Xmx2g"}) // Ajuste a memória conforme necessário
public class JMHBenchTest {

    // --- Início da duplicação controlada de campos estáticos para o escopo do benchmark ---
    // Para evitar problemas com estado estático global entre diferentes benchmarks ou execuções,
    // idealmente, a classe MovieRecommenderSGD seria refatorada para não usar campos estáticos
    // ou para permitir a passagem de instâncias de estado.
    // Como estamos trabalhando com a classe fornecida, vamos re-inicializar o estado estático
    // relevante no @Setup.
    private Map<String, double[]> userFactorsBenchmark;
    private Map<String, double[]> genreFactorsBenchmark;
    private java.util.Set<String> allGenresBenchmark;
    private java.util.Set<String> allUsersBenchmark;
    private Map<String, List<String>> movieToGenresMapBenchmark;
    // --- Fim da duplicação controlada ---

    private List<MovieRecommenderSGD.Rating> ratings;
    private Map<String, Map<String, Double>> predictedMatrix;

    // Caminho para o arquivo de dados. Ajuste se necessário.
    // Para um projeto Maven/Gradle, coloque em src/test/resources e carregue de forma diferente.
    // Por simplicidade, vamos assumir que está na raiz do projeto ou em um local fixo.
    private static final String SAMPLE_RATINGS_FILE = "dataset/avaliacoes_completas50MB.json";
    private static final String TEMP_OUTPUT_DIR = "jmh_output_ratings";
    private static final String TEMP_PREDICTED_RATINGS_FILE = TEMP_OUTPUT_DIR + "/predicted_ratings_benchmark.json";


    // Reinicializa os campos estáticos da classe MovieRecommenderSGD
    // Esta é uma forma de contornar o design com campos estáticos para o benchmark.
    private void resetStaticStateInMovieRecommenderSGD() {
        userFactorsBenchmark = new HashMap<>();
        genreFactorsBenchmark = new HashMap<>();
        allGenresBenchmark = new HashSet<>();
        allUsersBenchmark = new HashSet<>();
        movieToGenresMapBenchmark = new HashMap<>();

        // Atribuindo as instâncias do benchmark para os campos estáticos da classe original
        // Isso é arriscado e depende da visibilidade dos campos. Se forem privados, precisaria de reflexão ou getters/setters.
        // Assumindo que são package-private ou public para este exemplo.
        // Se MovieRecommenderSGD.userFactors etc. são privados, esta abordagem direta não funcionará
        // e você precisará refatorar MovieRecommenderSGD ou usar reflexão (não recomendado para benchmarks limpos).
        // Para este exemplo, vou assumir que podemos modificar a classe MovieRecommenderSGD
        // para ter setters para esses campos estáticos para fins de teste/benchmark, ou que eles são package-private.

        // MovieRecommenderSGD.userFactors = userFactorsBenchmark; // Exemplo, se fosse possível atribuir
        // MovieRecommenderSGD.genreFactors = genreFactorsBenchmark;
        // MovieRecommenderSGD.allGenres = allGenresBenchmark;
        // MovieRecommenderSGD.allUsers = allUsersBenchmark;

        // Alternativamente, e de forma mais realista se não pudermos mudar MovieRecommenderSGD:
        // Os métodos de MovieRecommenderSGD operarão em seus próprios campos estáticos.
        // O @Setup precisará popular esses campos estáticos diretamente.
        // E o @TearDown (ou @Setup no início de cada Trial) precisaria limpá-los.
        // Para simplificar aqui, vamos deixar os métodos estáticos de MovieRecommenderSGD
        // gerenciarem seus próprios campos estáticos como projetado, e o @Setup irá
        // popular `allUsersBenchmark` e `allGenresBenchmark` para uso local se necessário,
        // e então chamar os métodos de MovieRecommenderSGD que usam seus próprios estáticos.
    }


    @Setup(Level.Trial)
    public void setupTrial() throws IOException {
        System.out.println("Executando Setup Trial");

        File outputDir = new File(TEMP_OUTPUT_DIR);
        System.out.println("Tentando criar/verificar diretório em: " + outputDir.getAbsolutePath());
        if (!outputDir.exists()) {
            boolean created = outputDir.mkdirs();
            if (created) {
                System.out.println("Diretório criado com sucesso: " + outputDir.getAbsolutePath());
            } else {
                System.err.println("FALHA ao criar diretório: " + outputDir.getAbsolutePath());
                // Você pode querer lançar uma exceção aqui para interromper o benchmark se o diretório for crítico
                // throw new IOException("Não foi possível criar o diretório: " + outputDir.getAbsolutePath());
            }
        } else {
            System.out.println("Diretório já existe: " + outputDir.getAbsolutePath());
        }

        // Limpa e reinicializa os campos estáticos da MovieRecommenderSGD
        // (Idealmente, a classe MovieRecommenderSGD não teria estado estático mutável global)
        MovieRecommenderSGD.userFactors.clear();
        MovieRecommenderSGD.genreFactors.clear();
        MovieRecommenderSGD.allUsers.clear();
        MovieRecommenderSGD.allGenres.clear();
        // movieToGenresMap não é um campo estático em MovieRecommenderSGD, mas é usado.
        // Vamos recriá-lo aqui para o benchmark.
        movieToGenresMapBenchmark = new HashMap<>();


        ratings = MovieRecommenderSGD.loadRatings(SAMPLE_RATINGS_FILE);

        for (MovieRecommenderSGD.Rating r : ratings) {
            MovieRecommenderSGD.allUsers.add(r.getUserId()); // Popula o estático
            MovieRecommenderSGD.allGenres.addAll(r.getGenres()); // Popula o estático
            movieToGenresMapBenchmark.putIfAbsent(r.getTitle(), r.getGenres());
        }

        // Inicializa fatores para treino e predição
        MovieRecommenderSGD.initializeFactors(); // Usa e popula os campos estáticos de MovieRecommenderSGD

        // Treina o modelo uma vez para o benchmark de predição e salvamento
        MovieRecommenderSGD.trainModel(ratings);

        // Gera a matriz de predição uma vez para o benchmark de salvamento
        predictedMatrix = MovieRecommenderSGD.predictRatingsMatrix(ratings, movieToGenresMapBenchmark);
        System.out.println("Setup Trial concluído.");
    }

    @Setup(Level.Invocation)
    public void setupInvocation() {
        // Se precisarmos resetar algo específico entre cada chamada de um método de benchmark.
        // Por exemplo, para `benchmarkTrainModel`, gostaríamos de começar com fatores não treinados.
        // Então, `initializeFactors` deveria ser chamado aqui para `benchmarkTrainModel`.

        // Para `benchmarkInitializeFactors`, não precisamos de setup de invocação especial.
        // Para `benchmarkTrainModel`, precisamos que os fatores sejam inicializados, mas não treinados.
        // Para `benchmarkPredictRatingsMatrix`, precisamos que o modelo seja treinado.
        // Para `benchmarkSavePredictedRatingsToJson`, precisamos da matriz predita.

        // O Setup(Level.Trial) já lida com o estado inicial para a maioria dos benchmarks.
        // A exceção é `benchmarkTrainModel` e `benchmarkInitializeFactors` que modificam
        // o estado que outros benchmarks (ou execuções subsequentes do mesmo benchmark) podem usar.

        // Para `benchmarkInitializeFactors`: Limpar fatores antes de inicializar
        MovieRecommenderSGD.userFactors.clear();
        MovieRecommenderSGD.genreFactors.clear();
        // `allUsers` e `allGenres` já foram populados em `setupTrial` e `initializeFactors` os utiliza.

        // Para `benchmarkTrainModel`: Certificar que os fatores são recém-inicializados
        // (não já treinados pelo setupTrial ou outra execução)
        // Isso é complicado pelo fato de `initializeFactors` ser uma operação separada que queremos benchmarkar.
        // Vamos assumir para `benchmarkTrainModel` que `initializeFactors` é chamado separadamente
        // e que o benchmark mede apenas o `trainModel` sobre fatores já inicializados.
        // Para garantir que o treino comece do zero a cada invocação do benchmark de treino:
        // MovieRecommenderSGD.initializeFactors(); // Descomente se cada invocação de trainModel deve começar do zero.
        // Mas isso incluiria o tempo de initializeFactors no benchmark de trainModel.
        // Uma melhor abordagem seria ter uma cópia dos fatores inicializados
        // e restaurá-los aqui.
        // Por ora, `trainModel` continuará treinando sobre o estado atual dos fatores.
        // Isso significa que múltiplas invocações de `benchmarkTrainModel` dentro
        // da mesma Trial (se houver) continuarão o treinamento, o que pode ser
        // o comportamento desejado se você está medindo o custo de épocas adicionais.
        // Com @Fork(1) e @Warmup/@Measurement separados, cada "medição" efetiva
        // de `trainModel` começará a partir do estado deixado por `setupTrial`.
    }


    @Benchmark
    public void benchmarkLoadRatings(Blackhole bh) throws IOException {
        // É importante limpar os conjuntos estáticos se loadRatings for chamado múltiplas vezes
        // e se ele depender desses conjuntos estarem vazios ou em um estado específico.
        // No entanto, loadRatings em si não usa os campos estáticos da classe, apenas retorna uma lista.
        List<MovieRecommenderSGD.Rating> loadedRatings = MovieRecommenderSGD.loadRatings(SAMPLE_RATINGS_FILE);
        bh.consume(loadedRatings);
    }

    @Benchmark
    public void benchmarkInitializeFactors(Blackhole bh) {
        // `allUsers` e `allGenres` já devem estar populados por `setupTrial` via `loadRatings`.
        // `initializeFactors` usa esses conjuntos estáticos.
        // Para garantir que estamos medindo apenas `initializeFactors` sobre um estado consistente:
        MovieRecommenderSGD.userFactors.clear(); // Limpa o estado que será preenchido
        MovieRecommenderSGD.genreFactors.clear(); // Limpa o estado que será preenchido
        MovieRecommenderSGD.initializeFactors();
        bh.consume(MovieRecommenderSGD.userFactors); // Consome para evitar DCE
        bh.consume(MovieRecommenderSGD.genreFactors);
    }

    @Benchmark
    public void benchmarkTrainModel(Blackhole bh) {
        // Este benchmark medirá o treinamento a partir do estado atual dos fatores.
        // Se `setupInvocation` chamasse `initializeFactors()`, mediria `initialize + train`.
        // Como está, `setupTrial` chama `initializeFactors` e `trainModel` uma vez.
        // As invocações de `benchmarkTrainModel` irão *continuar* o treinamento ou
        // retrabalhar sobre fatores já parcialmente treinados se não houver reset.
        // Para medir o treino do zero, precisaríamos de um reset dos pesos em `setupInvocation`.
        // Vamos assumir que queremos medir o treino a partir de um estado já inicializado (feito no setupTrial).
        // Para um benchmark mais puro de "treinar do zero", copie os fatores iniciais e restaure-os em setupInvocation.
        // Ou, mais simples, chame initializeFactors() antes de trainModel() AQUI, aceitando que o tempo será combinado.

        // Para medir o `trainModel` sobre fatores "frescos" (apenas inicializados, não treinados):
        MovieRecommenderSGD.userFactors.clear();
        MovieRecommenderSGD.genreFactors.clear();
        MovieRecommenderSGD.initializeFactors(); // Garante que os fatores estão no estado inicial
        MovieRecommenderSGD.trainModel(ratings); // `ratings` foi carregado em setupTrial
        bh.consume(MovieRecommenderSGD.userFactors); // Consome para evitar DCE
    }

    @Benchmark
    public void benchmarkPredictRatingsMatrix(Blackhole bh) {
        // `userFactors` e `genreFactors` devem estar treinados (feito em `setupTrial`).
        // `allUsers` e `movieToGenresMapBenchmark` também de `setupTrial`.
        Map<String, Map<String, Double>> matrix = MovieRecommenderSGD.predictRatingsMatrix(ratings, movieToGenresMapBenchmark);
        bh.consume(matrix);
    }

    @Benchmark
    public void benchmarkSavePredictedRatingsToJson(Blackhole bh) throws IOException {
        // `predictedMatrix`, `MovieRecommenderSGD.allUsers`, `movieToGenresMapBenchmark` são de `setupTrial`.
        // Certifique-se de que o diretório de saída existe.
        File tempFile = new File(TEMP_PREDICTED_RATINGS_FILE);

        MovieRecommenderSGD.savePredictedRatingsToJson(predictedMatrix, MovieRecommenderSGD.allUsers, movieToGenresMapBenchmark, TEMP_PREDICTED_RATINGS_FILE);
        bh.consume(tempFile.length()); // Consome algo sobre o resultado da operação

        // Limpeza opcional do arquivo após cada medição para evitar problemas de disco/cache,
        // mas pode adicionar sobrecarga ao benchmark. JMH geralmente lida bem com isso.
        // Files.deleteIfExists(tempFile.toPath());
    }

    @TearDown(Level.Trial)
    public void tearDownTrial() throws IOException {
        System.out.println("Executando TearDown Trial");
        // Limpar arquivos temporários se necessário
        File tempOutputDir = new File(TEMP_OUTPUT_DIR);
        if (tempOutputDir.exists()) {
            Files.walk(tempOutputDir.toPath())
                    .sorted(java.util.Comparator.reverseOrder())
                    .map(java.nio.file.Path::toFile)
                    .forEach(File::delete);
        }
        // Não precisamos limpar o sample_ratings.json, ele pode ser reutilizado.
        System.out.println("TearDown Trial concluído.");
    }


    // Método Main para executar o benchmark (opcional, pode ser executado via API do JMH ou plugin Maven/Gradle)
    public static void main(String[] args) throws Exception {
        // Se MovieRecommenderSGD estiver em um JAR separado ou módulo, garanta que está no classpath.
        // Limpando a propriedade para garantir que o JMH use o caminho da classe corretamente
        System.setProperty("jmh.shutdown.builtSharedLibs", "false");

        org.openjdk.jmh.Main.main(args);

        // Alternativamente, para rodar programaticamente:
        // Options opt = new OptionsBuilder()
        //         .include(MovieRecommenderSGDBenchmark.class.getSimpleName())
        //         .forks(1)
        //         .jvmArgsAppend("-Xms2g", "-Xmx2g")
        //         .warmupIterations(3)
        //         .measurementIterations(5)
        //         .build();
        // new Runner(opt).run();
    }
}
