package SGD;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import com.google.gson.reflect.TypeToken;
import com.google.gson.stream.JsonWriter;
import org.jetbrains.annotations.NotNull;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MovieRecommenderSGD {

    // Definição da classe interna Rating
    private static class Rating {
        String user_id;
        String title;
        List<String> genres;
        double rating;

        // Construtor e getters
        public Rating(String user_id, String title, List<String> genres, double rating) {
            this.user_id = user_id;
            this.title = title;
            this.genres = genres;
            this.rating = rating;
        }

        public String getUserId() { return user_id; }
        public String getTitle() { return title; }
        public List<String> getGenres() { return genres; }
        public double getRating() { return rating; }
    }

    // Constantes do modelo
    private static final int NUM_FEATURES = 10;
    private static final double LEARNING_RATE = 0.01;
    private static final double REGULARIZATION = 0.02;
    private static final int NUM_EPOCHS = 100;

    // Estruturas de dados concorrentes para os fatores
    private static ConcurrentHashMap<String, double[]> userFactors = new ConcurrentHashMap<>();
    private static ConcurrentHashMap<String, double[]> genreFactors = new ConcurrentHashMap<>();
    private static Set<String> allUsers = ConcurrentHashMap.newKeySet();
    private static Set<String> allGenres = ConcurrentHashMap.newKeySet();

    // Utilitários para JSON
    private static final Gson gson = new Gson();
    private static final Type ratingListType = new TypeToken<List<Rating>>() {}.getType();

    // Carrega avaliações de múltiplos arquivos em paralelo usando Virtual Threads
    public static ConcurrentLinkedQueue<Rating> loadRatingsParallel(Set<String> filenames) {
        ConcurrentLinkedQueue<Rating> ratings = new ConcurrentLinkedQueue<>();
        try (ExecutorService virtualThreadExecutor = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<?>> futures = filenames.stream()
                    .map(filename -> virtualThreadExecutor.submit(() -> {
                        try (FileReader reader = new FileReader(filename)) {
                            List<Rating> localList = gson.fromJson(reader, ratingListType);
                            if (localList != null) {
                                ratings.addAll(localList);
                            }
                        } catch (IOException | JsonSyntaxException e) {
                            System.err.println("Erro ao processar arquivo: " + filename + " - " + e.getMessage());
                        }
                    }))
                    .collect(Collectors.toList());

            for (Future<?> f : futures) {
                try {
                    f.get(); // Espera a conclusão de cada tarefa
                } catch (InterruptedException | ExecutionException e) {
                    System.err.println("Erro ao aguardar carregamento de arquivo: " + e.getMessage());
                    if (e instanceof InterruptedException) Thread.currentThread().interrupt();
                }
            }
        }
        return ratings;
    }

    // Inicializa os vetores de fatores de forma paralela
    public static void initializeFactors(@NotNull ConcurrentLinkedQueue<Rating> ratings) throws InterruptedException {
        // Extrai usuários e gêneros únicos em paralelo
        ratings.parallelStream().forEach(r -> {
            allUsers.add(r.getUserId());
            allGenres.addAll(r.getGenres());
        });

        // Inicializa os fatores de usuários e gêneros em paralelo
        try (ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors())) {
            allUsers.forEach(user -> executor.submit(() -> {
                double[] features = ThreadLocalRandom.current().doubles(NUM_FEATURES, 0, 0.1).toArray();
                userFactors.put(user, features);
            }));

            allGenres.forEach(genre -> executor.submit(() -> {
                double[] features = ThreadLocalRandom.current().doubles(NUM_FEATURES, 0, 0.1).toArray();
                genreFactors.put(genre, features);
            }));

            executor.shutdown();
            executor.awaitTermination(1, TimeUnit.MINUTES);
        }
    }

    /**
     * Treina o modelo usando CompletableFuture para processamento paralelo de blocos de dados.
     * @param ratings A fila de avaliações para treinamento.
     */
    public static void trainModel(ConcurrentLinkedQueue<Rating> ratings) {
        System.out.println("Iniciando Treinamento com CompletableFuture.");
        List<Rating> ratingList = new ArrayList<>(ratings);
        int numThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        int chunkSize = (int) Math.ceil((double) ratingList.size() / (numThreads * 4));
        if (chunkSize == 0 && !ratingList.isEmpty()) chunkSize = 1;

        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            List<CompletableFuture<Void>> futures = new ArrayList<>();
            for (int i = 0; i < ratingList.size(); i += chunkSize) {
                final List<Rating> chunk = ratingList.subList(i, Math.min(i + chunkSize, ratingList.size()));

                CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                    for (Rating rating : chunk) {
                        String user = rating.getUserId();
                        double[] userVec = userFactors.get(user);
                        if (userVec == null) continue;

                        // Sincronização para evitar condição de corrida na atualização dos vetores
                        synchronized (userVec) {
                            for (String genre : rating.getGenres()) {
                                double[] genreVec = genreFactors.get(genre);
                                if (genreVec == null) continue;

                                double prediction = dot(userVec, genreVec);
                                double error = rating.getRating() - prediction;

                                synchronized (genreVec) {
                                    updateVectors(userVec, genreVec, error);
                                }
                            }
                        }
                    }
                }, executor);
                futures.add(future);
            }

            // Espera que todas as tarefas da época atual sejam concluídas
            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
        }

        executor.shutdown();
    }

    // Atualiza os vetores de características
    public static void updateVectors(double[] userVec, double[] genreVec, double error) {
        for (int i = 0; i < NUM_FEATURES; i++) {
            double u = userVec[i];
            double g = genreVec[i];
            userVec[i] += LEARNING_RATE * (error * g - REGULARIZATION * u);
            genreVec[i] += LEARNING_RATE * (error * u - REGULARIZATION * g);
        }
    }

    /**
     * Gera a matriz de predições usando CompletableFuture para processar usuários em paralelo.
     * @param ratings A fila de avaliações original.
     * @return Um mapa concorrente representando a matriz de predições.
     */
    private static ConcurrentMap<String, Map<String, Double>> predictRatingsMatrix(ConcurrentLinkedQueue<Rating> ratings) {
        System.out.println("Iniciando geração da matriz de predições com CompletableFuture.");
        // Pré-processa as avaliações para acesso rápido
        Map<String, List<Rating>> ratingsByUser = ratings.stream().collect(Collectors.groupingBy(Rating::getUserId));
        Map<String, List<String>> genresByTitle = ratings.stream()
                .collect(Collectors.toConcurrentMap(
                        Rating::getTitle,
                        r -> r.getGenres(),
                        (existing, replacement) -> existing // Mantém o primeiro encontrado
                ));

        ConcurrentMap<String, Map<String, Double>> matrix = new ConcurrentHashMap<>();
        int numThreads = Math.max(2, Runtime.getRuntime().availableProcessors());
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        List<String> userList = new ArrayList<>(allUsers);

        // Cria uma lista de CompletableFuture, um para cada usuário
        List<CompletableFuture<Void>> futures = userList.stream()
                .map(user -> CompletableFuture.runAsync(() -> {
                    double[] userVec = userFactors.get(user);
                    if (userVec == null) return;

                    Map<String, Double> userRatings = new ConcurrentHashMap<>();
                    // Adiciona avaliações existentes
                    List<Rating> knownRatings = ratingsByUser.get(user);
                    if (knownRatings != null) {
                        for (Rating r : knownRatings) {
                            userRatings.put(r.getTitle(), r.getRating());
                        }
                    }

                    // Prediz avaliações para filmes não avaliados
                    genresByTitle.forEach((title, genres) -> {
                        if (!userRatings.containsKey(title)) {
                            double predicted = 0.0;
                            int validGenres = 0;
                            for (String genre : genres) {
                                double[] genreVec = genreFactors.get(genre);
                                if (genreVec != null) {
                                    predicted += dot(userVec, genreVec);
                                    validGenres++;
                                }
                            }
                            if (validGenres > 0) {
                                userRatings.put(title, predicted / validGenres);
                            }
                        }
                    });
                    matrix.put(user, userRatings);
                }, executor))
                .collect(Collectors.toList());

        // Espera a conclusão de todas as predições
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();

        executor.shutdown();
        return matrix;
    }


    private static void savePredictedRatingsToJson(
            ConcurrentLinkedQueue<Rating> ratings,
            Map<String, Map<String, Double>> predictedMatrix,
            String outputDirectory,
            String baseFilename
    ) {
        List<String> allUsersOrdered = ratings.stream()
                .map(Rating::getUserId)
                .distinct()
                .sorted()
                .collect(Collectors.toList());

        if (allUsersOrdered.isEmpty() && !predictedMatrix.isEmpty()) {
            allUsersOrdered.addAll(predictedMatrix.keySet());
            Collections.sort(allUsersOrdered);
        }

        List<String> allTitlesOrdered = ratings.stream()
                .map(Rating::getTitle)
                .distinct()
                .sorted()
                .collect(Collectors.toList());

        if (allTitlesOrdered.isEmpty() && !predictedMatrix.isEmpty()) {
            Set<String> titlesSet = new HashSet<>();
            predictedMatrix.values().forEach(userMap -> titlesSet.addAll(userMap.keySet()));
            allTitlesOrdered.addAll(titlesSet);
            Collections.sort(allTitlesOrdered);
        }


        Map<String, List<String>> genreMap = new ConcurrentHashMap<>();
        if (ratings != null) {
            for (Rating r : ratings) {
                if (r != null && r.getTitle() != null && r.getGenres() != null) {
                    genreMap.putIfAbsent(r.getTitle(), r.getGenres());
                }
            }
        }



        List<Future<?>> futures = new ArrayList<>();

        int usersPerFileTarget = 15;
        int usersPerFile = allUsersOrdered.isEmpty() ? 1 : (int) Math.ceil((double) allUsersOrdered.size() / usersPerFileTarget);
        if (usersPerFile == 0 && !allUsersOrdered.isEmpty()) usersPerFile = 1;

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
                List<String> userChunk = allUsersOrdered.subList(i, end);

                String chunkFilename = String.format("%s_part_%d.json", baseFilename, filePart);
                String fullPath = outputDirectory + (outputDirectory.endsWith("/") ? "" : "/") + chunkFilename;

                final List<String> currentUserChunk = new ArrayList<>(userChunk);
                final Map<String, Map<String, Double>> finalPredictedMatrix = predictedMatrix;
                final List<String> finalAllTitlesOrdered = allTitlesOrdered;
                final Map<String, List<String>> finalGenreMap = genreMap;
                final String finalChunkFilename = chunkFilename;

                Future<?> future = virtualThreadExecutor.submit(() -> {

                    try (JsonWriter jsonWriter = new JsonWriter(new FileWriter(fullPath))) {
                        jsonWriter.setIndent("  ");

                        jsonWriter.beginArray();

                        for (String user : currentUserChunk) {
                            jsonWriter.beginObject();
                            jsonWriter.name("user_id").value(user);

                            jsonWriter.name("movies");
                            jsonWriter.beginArray();

                            Map<String, Double> userRatings = finalPredictedMatrix.getOrDefault(user, Collections.emptyMap());

                            for (String title : finalAllTitlesOrdered) {
                                jsonWriter.beginObject();
                                jsonWriter.name("title").value(title);

                                jsonWriter.name("genre");
                                jsonWriter.beginArray();
                                List<String> genres = finalGenreMap.getOrDefault(title, Collections.emptyList());
                                for (String genre : genres) {
                                    jsonWriter.value(genre);
                                }
                                jsonWriter.endArray();

                                jsonWriter.name("rating");
                                Double rating = userRatings.get(title);
                                if (rating != null) {
                                    jsonWriter.value(rating);
                                } else {
                                    jsonWriter.nullValue();
                                }
                                jsonWriter.endObject();
                            }
                            jsonWriter.endArray();
                            jsonWriter.endObject();
                        }
                        jsonWriter.endArray();

                    } catch (IOException e) {
                        System.err.println("Erro ao escrever arquivo " + finalChunkFilename + ": " + e.getMessage());
                        e.printStackTrace();
                    }
                });
                futures.add(future);
            }

            for (Future<?> f : futures) {
                try {
                    f.get();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    System.err.println("Thread principal interrompida enquanto aguardava o salvamento dos arquivos.");
                } catch (ExecutionException e) {
                    System.err.println("Erro na execução do salvamento de arquivo: " +
                            (e.getCause() != null ? e.getCause().getMessage() : e.getMessage()));

                }
            }
        }

        System.out.println("Processo de salvamento em múltiplos arquivos concluído.");
    }

    // Calcula o produto escalar entre dois vetores
    public static double dot(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();

        Set<String> arquivos = Set.of("../dataset/avaliacoes_completas50MB.json");
        ConcurrentLinkedQueue<Rating> ratings = loadRatingsParallel(arquivos);

        long readTime = System.nanoTime();
        System.out.printf("lidos em: %.2f segundos%n", (readTime - startTime) / 1e9);

        initializeFactors(ratings);
        long initialTime = System.nanoTime();
        System.out.printf("initializeFactors rodou em: %.2f segundos%n", (initialTime - readTime) / 1e9);

        trainModel(ratings);
        long trainingTime = System.nanoTime();
        System.out.printf("trainingModel rodou em: %.2f segundos%n", (trainingTime - initialTime) / 1e9);

        ConcurrentMap<String, Map<String, Double>> matrix = predictRatingsMatrix(ratings);
        long matrixTime = System.nanoTime();
        System.out.printf("matrixGen rodou em: %.2f segundos%n", (matrixTime - trainingTime) / 1e9);

        // Salvar os resultados
        // ...
        // --- Chamada da Função Modificada ---
        String outputDir = "output_ratings"; // Crie este diretório ou use um existente
        new java.io.File(outputDir).mkdirs(); // Garante que o diretório exista
        String baseFilename = "predicted_user_ratings";
        savePredictedRatingsToJson(ratings, matrix, outputDir, baseFilename);

        long endTime = System.nanoTime();
        System.out.printf("Tempo total: %.2f segundos%n", (endTime - startTime) / 1e9);
    }
}
