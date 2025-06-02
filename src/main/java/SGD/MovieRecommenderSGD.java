package SGD;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.google.gson.*;
import com.google.gson.annotations.SerializedName;
import com.google.gson.stream.JsonWriter; // Import necessário

import com.google.gson.reflect.TypeToken;
import org.jetbrains.annotations.NotNull;


public class MovieRecommenderSGD {

    private static class Rating {
        String user_id;
        String title;

        @SerializedName(value = "genres", alternate = "genre")
        List<String> genre;


        double rating;

        public Rating(String user_id, String title, List<String> genre, double rating) {
            this.user_id = user_id;
            this.title = title;
            this.genre = genre;
            this.rating = rating;
        }

        public String getUserId() {
            return user_id;
        }

        public String getTitle() {
            return title;
        }

        public List<String> getGenres() {
            return genre;
        }

        public double getRating() {
            return rating;
        }
    }


    // Classe RatingLoader (NÃO PRECISA DE MUDANÇAS AQUI, pois usa a definição de Rating)
    private static class RatingLoader {
        private static final Gson gson = new Gson();
        private static final Type ratingListType = new TypeToken<List<Rating>>() {}.getType();

        private static ConcurrentLinkedQueue<Rating> loadRatingsParallel(Set<String> filenames) {
            ConcurrentLinkedQueue<Rating> ratings = new ConcurrentLinkedQueue<>();
            List<Future<?>> futures = new ArrayList<>();
            try (ExecutorService virtualThreadExecutor = Executors.newVirtualThreadPerTaskExecutor()) {
                for (String filename : filenames) {
                    Future<?> future = virtualThreadExecutor.submit(() -> {
                        try (FileReader reader = new FileReader(filename)) {
                            // Gson usará a definição atualizada da classe Rating para o parsing
                            List<Rating> localList = gson.fromJson(reader, ratingListType);
                            if (localList != null) {
                                ratings.addAll(localList);
                            }
                        } catch (IOException e) {
                            System.err.println("Erro ao ler arquivo: " + filename + " (" + e.getClass().getSimpleName() + ": " + e.getMessage() + ")");
                        } catch (JsonSyntaxException e) {
                            System.err.println("Erro de sintaxe JSON no arquivo: " + filename + " (" + e.getMessage() + ")");
                        }
                    });
                    futures.add(future);
                }
                for (Future<?> f : futures) {
                    try {
                        f.get();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        System.err.println("Carregamento de ratings interrompido.");
                    } catch (ExecutionException e) {
                        System.err.println("Erro durante execução do carregamento de arquivo: " + e.getCause());
                    }
                }
            }
            return ratings;
        }
        private static ConcurrentLinkedQueue<Rating> loadRatingsParallel(String filename) {
            return loadRatingsParallel(Set.of(filename));
        }
    }

    private static final int NUM_FEATURES = 10;
    private static final double LEARNING_RATE = 0.01;
    private static final double REGULARIZATION = 0.02;
    private static final int NUM_EPOCHS = 100;

    private static ConcurrentHashMap<String, double[]> userFactors = new ConcurrentHashMap<>();
    private static ConcurrentHashMap<String, double[]> genreFactors = new ConcurrentHashMap<>();
    private static Set<String> allGenres = ConcurrentHashMap.newKeySet();
    private static Set<String> allUsers = ConcurrentHashMap.newKeySet();




    public static Set<String> carregarArquivosDaPasta(String caminhoDaPasta) throws IOException {
        System.out.println("Lendo arquivos da pasta: " + caminhoDaPasta);
        try (Stream<Path> streamDePaths = Files.list(Paths.get(caminhoDaPasta))) {
            Set<String> arquivosEncontrados = streamDePaths
                    .filter(Files::isRegularFile) // Garante que estamos pegando apenas arquivos
                    .peek(path -> System.out.println("Encontrado arquivo: " + path.toString())) // Opcional: para logar os arquivos encontrados
                    .map(Path::toString)          // Converte o Path para String
                    .collect(Collectors.toSet()); // Coleta os resultados em um Set
            if (arquivosEncontrados.isEmpty()) {
                System.out.println("Nenhum arquivo encontrado na pasta: " + caminhoDaPasta);
            }
            return arquivosEncontrados;
        }
    }


    public static void main(String[] args) throws Exception {
        // Substitua a linha original pela chamada da função
        Set<String> arquivos;
        /*
        String pastaDeDatasets = "dataset/avaliacao_individual"; // Defina o nome da sua pasta aqui
        try {
            arquivos = carregarArquivosDaPasta(pastaDeDatasets);
            if (arquivos.isEmpty()) {
                System.err.println("Nenhum arquivo encontrado em '" + pastaDeDatasets + "'. Verifique o caminho e o conteúdo da pasta.");
                return; // Encerra se nenhum arquivo for encontrado para evitar erros subsequentes
            }
        } catch (IOException e) {
            System.err.println("Erro ao ler arquivos da pasta '" + pastaDeDatasets + "': " + e.getMessage());
            // e.printStackTrace(); // Descomente para mais detalhes do erro
            return; // Encerra em caso de erro de leitura da pasta
        }
        */

        arquivos = Set.of("dataset/avaliacoes_completas50MB.json");


        long startTime = System.nanoTime();

        ConcurrentLinkedQueue<Rating> ratings = RatingLoader.loadRatingsParallel(arquivos);

        long readTime = System.nanoTime();
        System.out.printf("lidos em: %.2f segundos%n", (readTime - startTime) / 1e9);

        // Certifique-se de que 'ratings' não está vazio antes de prosseguir
        if (ratings.isEmpty() && !arquivos.isEmpty()) {
            System.err.println("Apesar dos arquivos serem listados, nenhum rating foi carregado. Verifique o formato dos arquivos e a lógica de RatingLoader.loadRatingsParallel.");
            return;
        } else if (ratings.isEmpty()) {
            // Mensagem já dada acima, mas podemos reforçar.
            System.err.println("Nenhum rating para processar.");
            return;
        }

        //saveOriginalMatrixWithNulls(ratings, "dataset/avaliacoes_iniciais_com_nulls.json");
        initializeFactors(ratings);

        long initialTime = System.nanoTime();
        System.out.printf("initializeFactors rodou em: %.2f segundos%n", (initialTime - readTime) / 1e9);




        trainModel(ratings);

        long trainingTime = System.nanoTime();
        System.out.printf("trainingModel rodou em: %.2f segundos%n", (trainingTime - initialTime) / 1e9);


        ConcurrentMap<String, Map<String, Double>> matrix = predictRatingsMatrix(ratings);

        long matrixTime = System.nanoTime();
        System.out.printf("matrixGen rodou em: %.2f segundos%n", (matrixTime - trainingTime) / 1e9);


        //printRatingsMatrix(matrix, ratings);

        long printTime = System.nanoTime();
        //  System.out.printf("printMatrix rodou em: %.2f segundos%n", (printTime - matrixTime) / 1e9);

        // --- Chamada da Função Modificada ---
        String outputDir = "output_ratings"; // Crie este diretório ou use um existente
        new java.io.File(outputDir).mkdirs(); // Garante que o diretório exista
        String baseFilename = "predicted_user_ratings";


        savePredictedRatingsToJson(ratings, matrix, outputDir, baseFilename);


        long saveTime = System.nanoTime();
        System.out.printf("saveJson rodou em: %.2f segundos%n", (saveTime - printTime) / 1e9);


        long endTime = System.nanoTime();
        System.out.printf("Tempo total: %.2f segundos%n", (endTime - startTime) / 1e9);

    }


    private static void initializeFactors(@NotNull ConcurrentLinkedQueue<Rating> ratings) throws InterruptedException {
        allUsers = ConcurrentHashMap.newKeySet();
        allGenres = ConcurrentHashMap.newKeySet();

        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {

            // Popular allUsers e allGenres em paralelo
            for (Rating r : ratings) {
                executor.submit(() -> {
                    allUsers.add(r.getUserId());
                    allGenres.addAll(r.getGenres());
                });
            }

            executor.shutdown();
            executor.awaitTermination(1, TimeUnit.MINUTES);
        }

        // Inicializar userFactors e genreFactors em paralelo, usando ThreadLocalRandom
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {

            for (String user : allUsers) {
                executor.submit(() -> {
                    double[] features = ThreadLocalRandom.current()
                            .doubles(NUM_FEATURES, 0, 0.1)
                            .toArray();
                    userFactors.put(user, features);
                });
            }

            for (String genre : allGenres) {
                executor.submit(() -> {
                    double[] features = ThreadLocalRandom.current()
                            .doubles(NUM_FEATURES, 0, 0.1)
                            .toArray();
                    genreFactors.put(genre, features);
                });
            }

            executor.shutdown();
            executor.awaitTermination(1, TimeUnit.MINUTES);
        }
    }



    private static void trainModel(ConcurrentLinkedQueue<Rating> ratings) throws InterruptedException {
        System.out.println("Iniciando Treinamento");
        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
                for (Rating r : ratings) {
                    executor.submit(() -> {
                        String user = r.getUserId();
                        double[] userVec = userFactors.get(user);
                        for (String genre : r.getGenres()) {
                            double[] genreVec = genreFactors.get(genre);
                            double prediction = dot(userVec, genreVec);
                            double error = r.getRating() - prediction;

                            for (int i = 0; i < NUM_FEATURES; i++) {
                                double u = userVec[i];
                                double g = genreVec[i];
                                userVec[i] += LEARNING_RATE * (error * g - REGULARIZATION * u);
                                genreVec[i] += LEARNING_RATE * (error * u - REGULARIZATION * g);
                            }
                        }
                    });
                }
                executor.shutdown();
                executor.awaitTermination(5, TimeUnit.MINUTES);
            }
        }
    }


    private static ConcurrentMap<String, Map<String, Double>> predictRatingsMatrix(ConcurrentLinkedQueue<Rating> ratings) {
        ConcurrentMap<String, ConcurrentLinkedQueue<Rating>> ratingsByUser = new ConcurrentHashMap<>();
        for (Rating r : ratings) {
            ratingsByUser
                    .computeIfAbsent(r.getUserId(), k -> new ConcurrentLinkedQueue<>())
                    .add(r);
        }


        ConcurrentMap<String, Map<String, Double>> matrix = new ConcurrentHashMap<>();
        List<Thread> threads = new ArrayList<>();

        for (String user : allUsers) {
            Thread t = Thread.ofVirtual().start(() -> {
                Map<String, Double> ratingsForUser = new ConcurrentHashMap<>();
                ConcurrentLinkedQueue<Rating> userRatings = ratingsByUser.getOrDefault(user, new ConcurrentLinkedQueue<>());

                for (Rating r : userRatings) {
                    ratingsForUser.put(r.getTitle(), r.getRating());
                }

                for (Rating r : ratings) {
                    if (!ratingsForUser.containsKey(r.getTitle())) {
                        double predicted = 0.0;
                        for (String genre : r.getGenres()) {
                            predicted += dot(userFactors.get(user), genreFactors.get(genre));
                        }
                        predicted /= r.getGenres().size();
                        ratingsForUser.put(r.getTitle(), predicted);
                    }
                }

                matrix.put(user, ratingsForUser);
            });
            threads.add(t);
        }

        for (Thread t : threads) {
            try {
                t.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        return matrix;
    }


    private static void savePredictedRatingsToJson(
            ConcurrentLinkedQueue<Rating> ratings, // Fila de avaliações original
            Map<String, Map<String, Double>> predictedMatrix, // Matriz de predições
            String outputDirectory, // Diretório de saída
            String baseFilename // Nome base para os arquivos de saída
    ) {
        // 1. Obter listas ordenadas de todos os usuários e títulos
        List<String> allUsersOrdered = ratings.stream()
                .map(Rating::getUserId)
                .distinct()
                .sorted()
                .collect(Collectors.toList());

        // Se a lista de ratings estiver vazia, mas a predictedMatrix não, tenta obter usuários da matriz
        if (allUsersOrdered.isEmpty() && !predictedMatrix.isEmpty()) {
            allUsersOrdered.addAll(predictedMatrix.keySet());
            Collections.sort(allUsersOrdered); // Garante a ordem
        }

        List<String> allTitlesOrdered = ratings.stream()
                .map(Rating::getTitle)
                .distinct()
                .sorted()
                .collect(Collectors.toList());

        // Se a lista de ratings estiver vazia, mas a predictedMatrix não, tenta obter títulos da matriz
        if (allTitlesOrdered.isEmpty() && !predictedMatrix.isEmpty()) {
            Set<String> titlesSet = new HashSet<>();
            predictedMatrix.values().forEach(userMap -> titlesSet.addAll(userMap.keySet()));
            allTitlesOrdered.addAll(titlesSet);
            Collections.sort(allTitlesOrdered); // Garante a ordem
        }

        // 2. Criar um mapa de títulos para gêneros
        // Usar ConcurrentHashMap é seguro, mas como é preenchido sequencialmente antes das threads,
        // um HashMap normal também funcionaria se não modificado depois.
        Map<String, List<String>> genreMap = new ConcurrentHashMap<>();
        if (ratings != null) { // Adiciona verificação de nulo para 'ratings'
            for (Rating r : ratings) {
                if (r != null && r.getTitle() != null && r.getGenres() != null) { // Verifica r, título e gêneros
                    genreMap.putIfAbsent(r.getTitle(), r.getGenres());
                }
            }
        }


        // Gson ainda pode ser útil para configurar o JsonWriter ou para outras tarefas, mas não para toJson em si.
        // Gson gson = new GsonBuilder().setPrettyPrinting().create(); // Não é mais usado para a serialização principal

        List<Future<?>> futures = new ArrayList<>();

        // Lógica para dividir usuários em arquivos
        int usersPerFileTarget = 15; // Seu valor original, interpretado como "objetivo de arquivos"
        int usersPerFile = allUsersOrdered.isEmpty() ? 1 : (int) Math.ceil((double) allUsersOrdered.size() / usersPerFileTarget);
        if (usersPerFile == 0 && !allUsersOrdered.isEmpty()) usersPerFile = 1; // Garante pelo menos 1 se houver usuários

        System.out.printf("Iniciando salvamento de %d usuários em blocos de aproximadamente %d usuários por arquivo (total %d arquivos).%n",
                allUsersOrdered.size(),
                usersPerFile,
                allUsersOrdered.isEmpty() ? 0 : (int) Math.ceil((double) allUsersOrdered.size() / usersPerFile));

        try (ExecutorService virtualThreadExecutor = Executors.newVirtualThreadPerTaskExecutor()) {
            int filePart = 0;
            if (allUsersOrdered.isEmpty()) {
                System.out.println("Nenhum usuário para salvar.");
            }

            for (int i = 0; i < allUsersOrdered.size(); i += usersPerFile) {
                filePart++;
                int end = Math.min(i + usersPerFile, allUsersOrdered.size());
                List<String> userChunk = allUsersOrdered.subList(i, end); // Sublista para o chunk atual

                // Nome final do arquivo para este chunk
                String chunkFilename = String.format("%s_part_%d.json", baseFilename, filePart);
                String fullPath = outputDirectory + (outputDirectory.endsWith("/") ? "" : "/") + chunkFilename;

                // Copia o chunk de usuários para garantir que seja efetivamente final para o lambda
                final List<String> currentUserChunk = new ArrayList<>(userChunk);
                final Map<String, Map<String, Double>> finalPredictedMatrix = predictedMatrix;
                final List<String> finalAllTitlesOrdered = allTitlesOrdered;
                final Map<String, List<String>> finalGenreMap = genreMap;
                final String finalChunkFilename = chunkFilename; // Para mensagem de erro

                Future<?> future = virtualThreadExecutor.submit(() -> {
                    // NÃO construir List<Map<String, Object>> usersOutputForChunk em memória
                    // Em vez disso, usar JsonWriter para streaming direto para o arquivo.
                    try (JsonWriter jsonWriter = new JsonWriter(new FileWriter(fullPath))) {
                        jsonWriter.setIndent("  "); // Para "pretty printing"

                        jsonWriter.beginArray(); // Cada arquivo de chunk será um array de usuários

                        for (String user : currentUserChunk) {
                            jsonWriter.beginObject(); // Inicia objeto do usuário: {
                            jsonWriter.name("user_id").value(user);

                            jsonWriter.name("movies");
                            jsonWriter.beginArray(); // Inicia lista de filmes para este usuário: [

                            Map<String, Double> userRatings = finalPredictedMatrix.getOrDefault(user, Collections.emptyMap());

                            for (String title : finalAllTitlesOrdered) {
                                jsonWriter.beginObject(); // Inicia dados do filme: {
                                jsonWriter.name("title").value(title);

                                jsonWriter.name("genre");
                                jsonWriter.beginArray(); // Inicia lista de gêneros para este filme: [
                                List<String> genres = finalGenreMap.getOrDefault(title, Collections.emptyList());
                                for (String genre : genres) {
                                    jsonWriter.value(genre);
                                }
                                jsonWriter.endArray(); // Finaliza lista de gêneros: ]

                                jsonWriter.name("rating");
                                Double rating = userRatings.get(title);
                                if (rating != null) {
                                    jsonWriter.value(rating); // Escreve o número de avaliação
                                } else {
                                    jsonWriter.nullValue(); // Escreve null JSON se a avaliação não existir
                                }
                                jsonWriter.endObject(); // Finaliza dados do filme: }
                            }
                            jsonWriter.endArray(); // Finaliza lista de filmes para este usuário: ]
                            jsonWriter.endObject(); // Finaliza objeto do usuário: }
                        }
                        jsonWriter.endArray(); // Finaliza o array de usuários para este arquivo de chunk: ]

                    } catch (IOException e) {
                        System.err.println("Erro ao escrever arquivo " + finalChunkFilename + ": " + e.getMessage());
                        // Você pode querer lançar uma RuntimeException aqui ou usar um manipulador de exceção mais robusto
                        // e.printStackTrace(); // Para depuração
                    }
                });
                futures.add(future);
            }

            // Aguarda a conclusão de todas as tarefas de salvamento
            for (Future<?> f : futures) {
                try {
                    f.get(); // Espera a conclusão e obtém exceções, se houver
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt(); // Restaura o status de interrupção
                    System.err.println("Thread principal interrompida enquanto aguardava o salvamento dos arquivos.");
                } catch (ExecutionException e) {
                    System.err.println("Erro na execução do salvamento de arquivo: " +
                            (e.getCause() != null ? e.getCause().getMessage() : e.getMessage()));
                    // e.printStackTrace(); // Para depuração da causa raiz
                }
            }
        } // O ExecutorService é fechado automaticamente aqui (try-with-resources)

        System.out.println("Processo de salvamento em múltiplos arquivos concluído.");
    }




    private static double dot(double[] a, double[] b) {
        final int length = a.length;
        double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
        int i = 0;
        for (; i <= length - 4; i += 4) {
            sum0 += a[i] * b[i];
            sum1 += a[i + 1] * b[i + 1];
            sum2 += a[i + 2] * b[i + 2];
            sum3 += a[i + 3] * b[i + 3];
        }
        double sum = sum0 + sum1 + sum2 + sum3;
        for (; i < length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

}