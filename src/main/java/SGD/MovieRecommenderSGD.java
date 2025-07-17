package SGD;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import com.google.gson.*;
import com.google.gson.reflect.TypeToken;
import com.google.gson.stream.JsonWriter;
import org.jetbrains.annotations.NotNull;


public class MovieRecommenderSGD {

    private static class Rating {
        String user_id;
        String title;
        List<String> genres;
        double rating;

        public Rating(String user_id, String title, List<String> genres, double rating) {
            this.user_id = user_id;
            this.title = title;
            this.genres = genres;
            this.rating = rating;
        }

        public String getUserId() {
            return user_id;
        }

        public String getTitle() {
            return title;
        }

        public List<String> getGenres() {
            return genres;
        }

        public double getRating() {
            return rating;
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


    private static final Gson gson = new Gson();
    private static final Type ratingListType = new TypeToken<List<Rating>>() {
    }.getType();

    public static ConcurrentLinkedQueue<Rating> loadRatingsParallel(Set<String> filenames) {
        ConcurrentLinkedQueue<Rating> ratings = new ConcurrentLinkedQueue<>();
        List<Future<?>> futures = new ArrayList<>();

        try (ExecutorService virtualThreadExecutor = Executors.newVirtualThreadPerTaskExecutor()) {
            for (String filename : filenames) {
                Future<?> future = virtualThreadExecutor.submit(() -> {
                    try (FileReader reader = new FileReader(filename)) {
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

    public static ConcurrentLinkedQueue<Rating> loadRatingsParallel(String filename) {
        return loadRatingsParallel(Set.of(filename));
    }


    public static void initializeFactors(@NotNull ConcurrentLinkedQueue<Rating> ratings) throws InterruptedException {
        allUsers = ConcurrentHashMap.newKeySet();
        allGenres = ConcurrentHashMap.newKeySet();

        ExecutorService executor1 = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        for (Rating r : ratings) {
            executor1.submit(() -> {
                allUsers.add(r.getUserId());
                allGenres.addAll(r.getGenres());
            });
        }

        executor1.shutdown();
        executor1.awaitTermination(1, TimeUnit.MINUTES);

        ExecutorService executor2 = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        for (String user : allUsers) {
            executor2.submit(() -> {
                double[] features = ThreadLocalRandom.current()
                        .doubles(NUM_FEATURES, 0, 0.1)
                        .toArray();
                userFactors.put(user, features);
            });
        }

        for (String genre : allGenres) {
            executor2.submit(() -> {
                double[] features = ThreadLocalRandom.current()
                        .doubles(NUM_FEATURES, 0, 0.1)
                        .toArray();
                genreFactors.put(genre, features);
            });
        }

        executor2.shutdown();
        executor2.awaitTermination(1, TimeUnit.MINUTES);
    }


    public static void trainModel(ConcurrentLinkedQueue<Rating> ratings) {
        System.out.println("Iniciando Treinamento com Parallel Streams.");
        List<Rating> ratingList = new ArrayList<>(ratings);

        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            ratingList.parallelStream().forEach(rating -> {
                String user = rating.getUserId();
                double[] userVec = userFactors.get(user);

                if (userVec == null) return;

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
            });
        }
        System.out.println("Treinamento concluído.");
    }


    public static void updateVectors(double[] userVec, double[] genreVec, double error) {
        for (int i = 0; i < NUM_FEATURES; i++) {
            double u = userVec[i];
            double g = genreVec[i];
            userVec[i] += LEARNING_RATE * (error * g - REGULARIZATION * u);
            genreVec[i] += LEARNING_RATE * (error * u - REGULARIZATION * g);
        }
    }


    private static ConcurrentMap<String, Map<String, Double>> predictRatingsMatrix(ConcurrentLinkedQueue<Rating> ratings) {

        final Map<String, List<Rating>> ratingsByUser = ratings.stream()
                .collect(Collectors.groupingBy(Rating::getUserId));
        final Map<String, List<Rating>> ratingsByTitle = ratings.stream()
                .collect(Collectors.groupingBy(Rating::getTitle));


        return allUsers.parallelStream().collect(Collectors.toConcurrentMap(
                user -> user,
                user -> {
                    double[] userVec = userFactors.get(user);
                    if (userVec == null) {
                        return Collections.emptyMap();
                    }

                    Map<String, Double> userRatings = new HashMap<>();
                    List<Rating> knownRatings = ratingsByUser.get(user);
                    if (knownRatings != null) {
                        for (Rating r : knownRatings) {
                            userRatings.put(r.getTitle(), r.getRating());
                        }
                    }

                    for (Map.Entry<String, List<Rating>> entry : ratingsByTitle.entrySet()) {
                        String title = entry.getKey();
                        if (userRatings.containsKey(title)) {
                            continue;
                        }

                        List<Rating> ratingsForTitle = entry.getValue();
                        if (ratingsForTitle.isEmpty()) continue;

                        List<String> currentGenres = ratingsForTitle.get(0).getGenres();
                        if (currentGenres == null || currentGenres.isEmpty()) continue;

                        double predicted = 0.0;
                        int validGenres = 0;
                        for (String genre : currentGenres) {
                            double[] genreVec = genreFactors.get(genre);
                            if (genreVec != null) {
                                predicted += dot(userVec, genreVec);
                                validGenres++;
                            }
                        }

                        if (validGenres > 0) {
                            predicted /= validGenres;
                            userRatings.put(title, predicted);
                        }
                    }
                    return userRatings;
                }
        ));
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



    public static double dot(double[] a, double[] b) {
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

    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();

        //Set<String> arquivos = Set.of("C:\\Users\\Suporte\\Documents\\vinicius\\Programação Distribuída\\MF-SGD\\dataset\\avaliacoes_completas100MB.json");

        Set<String> arquivos = Set.of("../dataset/avaliacoes_completas50MB.json");

        ConcurrentLinkedQueue<Rating> ratings = loadRatingsParallel(arquivos);

        long readTime = System.nanoTime();
        System.out.printf("lidos em: %.2f segundos%n", (readTime - startTime) / 1e9);

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

        String outputDir = "output_ratings";
        new java.io.File(outputDir).mkdirs();
        String baseFilename = "predicted_user_ratings";


        savePredictedRatingsToJson(ratings, matrix, outputDir, baseFilename);


        long saveTime = System.nanoTime();
        System.out.printf("saveJson rodou em: %.2f segundos%n", (saveTime - printTime) / 1e9);


        long endTime = System.nanoTime();
        System.out.printf("Tempo total: %.2f segundos%n", (endTime - startTime) / 1e9);
    }

}