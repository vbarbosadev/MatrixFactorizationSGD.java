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
import org.jetbrains.annotations.NotNull;


public class MovieRecommenderSGDvPlatform {

    private static class Rating {
        String user_id;
        String title;
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

    private static class RatingLoader {

        private static final Gson gson = new Gson();
        private static final Type ratingListType = new TypeToken<List<Rating>>() {}.getType();

        public static ConcurrentLinkedQueue<Rating> loadRatingsParallel(Set<String> filenames) {
            ConcurrentLinkedQueue<Rating> ratings = new ConcurrentLinkedQueue<>();
            List<Thread> threads = new ArrayList<>();

            for (String filename : filenames) {
                Thread t = new Thread(() -> {
                    try (FileReader reader = new FileReader(filename)) {
                        List<Rating> localList = gson.fromJson(reader, ratingListType);
                        ratings.addAll(localList);
                    } catch (IOException e) {
                        System.err.println("Erro ao ler arquivo: " + filename);
                        e.printStackTrace();
                    }
                });
                t.start(); // Lembre-se: platform threads precisam ser iniciadas manualmente
                threads.add(t);
            }

            for (Thread t : threads) {
                try {
                    t.join();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }

            return ratings;
        }

        // Sobrecarga para um único arquivo
        public static ConcurrentLinkedQueue<Rating> loadRatingsParallel(String filename) {
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


    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();

        Set<String> arquivos = Set.of("dataset/ratings_20MB.json");

        ConcurrentLinkedQueue<Rating> ratings = RatingLoader.loadRatingsParallel(arquivos);

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

        savePredictedRatingsToJson(ratings, matrix, "dataset/predicted_ratingsPlatform.json");

        long saveTime = System.nanoTime();
        System.out.printf("saveJson rodou em: %.2f segundos%n", (saveTime - printTime) / 1e9);


        long endTime = System.nanoTime();
        System.out.printf("Tempo total: %.2f segundos%n", (endTime - startTime) / 1e9);
    }


    private static void initializeFactors(@NotNull ConcurrentLinkedQueue<Rating> ratings) throws InterruptedException {
        allUsers = ConcurrentHashMap.newKeySet();
        allGenres = ConcurrentHashMap.newKeySet();

        // Etapa 1: Popular allUsers e allGenres
        ExecutorService executor1 = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        for (Rating r : ratings) {
            executor1.submit(() -> {
                allUsers.add(r.getUserId());
                allGenres.addAll(r.getGenres());
            });
        }

        executor1.shutdown();
        executor1.awaitTermination(1, TimeUnit.MINUTES);

        // Etapa 2: Inicializar userFactors e genreFactors
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

    private static void trainModel(ConcurrentLinkedQueue<Rating> ratings) {
        System.out.println("Iniciando Treinamento (pronto para sincronização futura)");

        List<Rating> ratingList = new ArrayList<>(ratings);

        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            ratingList.parallelStream().forEach(rating -> {
                String user = rating.getUserId();
                double[] userVec = userFactors.get(user);

                for (String genre : rating.getGenres()) {
                    double[] genreVec = genreFactors.get(genre);

                    double prediction = dot(userVec, genreVec);
                    double error = rating.getRating() - prediction;

                    // ⛔ Se quiser sincronizar depois:
                    // synchronized (getUserLock(user)) { ... }
                    // synchronized (getGenreLock(genre)) { ... }
                    updateVectors(userVec, genreVec, error);
                }
            });
        }
    }

    private static void updateVectors(double[] userVec, double[] genreVec, double error) {
        for (int i = 0; i < NUM_FEATURES; i++) {
            double u = userVec[i];
            double g = genreVec[i];
            userVec[i] += LEARNING_RATE * (error * g - REGULARIZATION * u);
            genreVec[i] += LEARNING_RATE * (error * u - REGULARIZATION * g);
        }
    }


    private static ConcurrentMap<String, Map<String, Double>> predictRatingsMatrix(ConcurrentLinkedQueue<Rating> ratings)
            throws InterruptedException {

        // Agrupar ratings por usuário
        ConcurrentMap<String, List<Rating>> ratingsByUser = new ConcurrentHashMap<>();
        for (Rating r : ratings) {
            ratingsByUser
                    .computeIfAbsent(r.getUserId(), k -> new ArrayList<>())
                    .add(r);
        }

        // Lista final com predições
        ConcurrentMap<String, Map<String, Double>> matrix = new ConcurrentHashMap<>();

        // Configuração da thread pool com Platform Threads
        int numThreads = Math.max(2, Runtime.getRuntime().availableProcessors());
        ExecutorService executor = Executors.newFixedThreadPool(numThreads, Thread.ofPlatform().factory());

        // Converter ratings para List para acesso mais rápido
        List<Rating> allRatings = new ArrayList<>(ratings);
        List<String> userList = new ArrayList<>(allUsers);

        // Tarefas em chunk por usuário (reduz número de tasks)
        int chunkSize = (int) Math.ceil(userList.size() / (double) numThreads);
        List<Callable<Void>> tasks = new ArrayList<>();

        for (int i = 0; i < userList.size(); i += chunkSize) {
            int start = i;
            int end = Math.min(i + chunkSize, userList.size());
            List<String> chunk = userList.subList(start, end);

            tasks.add(() -> {
                for (String user : chunk) {
                    Map<String, Double> userRatings = new HashMap<>();

                    // Ratings reais do usuário
                    List<Rating> knownRatings = ratingsByUser.getOrDefault(user, List.of());
                    for (Rating r : knownRatings) {
                        userRatings.put(r.getTitle(), r.getRating());
                    }

                    double[] userVec = userFactors.get(user);
                    if (userVec == null) continue;

                    // Predizer apenas filmes não avaliados por este usuário
                    for (Rating r : allRatings) {
                        String title = r.getTitle();
                        if (userRatings.containsKey(title)) continue;

                        List<String> genres = r.getGenres();
                        if (genres.isEmpty()) continue;

                        double predicted = 0.0;
                        for (String genre : genres) {
                            double[] genreVec = genreFactors.get(genre);
                            if (genreVec != null) {
                                predicted += dot(userVec, genreVec);
                            }
                        }

                        predicted /= genres.size();
                        userRatings.put(title, predicted);
                    }

                    matrix.put(user, userRatings);
                }
                return null;
            });
        }

        // Executa tarefas
        executor.invokeAll(tasks);
        executor.shutdown();
        executor.awaitTermination(30, TimeUnit.MINUTES);

        return matrix;
    }


    private static void printRatingsMatrix(ConcurrentMap<String, Map<String, Double>> matrix,
                                           ConcurrentLinkedQueue<Rating> ratings) throws InterruptedException {
        List<String> movies = ratings.stream()
                .map(Rating::getTitle)
                .distinct()
                .collect(Collectors.toList());

        System.out.print("Usuário\t");
        for (String movie : movies) {
            System.out.print(movie + "\t");
        }
        System.out.println();

        ExecutorService executor = Executors.newFixedThreadPool(
                Math.max(2, Runtime.getRuntime().availableProcessors()), Thread.ofPlatform().factory());

        ConcurrentLinkedQueue<String> outputLines = new ConcurrentLinkedQueue<>();
        List<Callable<Void>> tasks = new ArrayList<>();

        for (String user : allUsers) {
            tasks.add(() -> {
                StringBuilder sb = new StringBuilder();
                sb.append(user).append("\t");

                Map<String, Double> userRatings = matrix.getOrDefault(user, Map.of());
                for (String movie : movies) {
                    sb.append(String.format("%.2f\t", userRatings.getOrDefault(movie, 0.0)));
                }

                outputLines.add(sb.toString());
                return null;
            });
        }

        executor.invokeAll(tasks);
        executor.shutdown();

        outputLines.forEach(System.out::println);
    }






    private static void savePredictedRatingsToJson(
            ConcurrentLinkedQueue<Rating> ratings,
            Map<String, Map<String, Double>> predictedMatrix,
            String filename
    ) throws InterruptedException {
        List<String> allUsersOrdered = ratings.stream()
                .map(Rating::getUserId)
                .distinct()
                .sorted()
                .toList();

        List<String> allTitlesOrdered = ratings.stream()
                .map(Rating::getTitle)
                .distinct()
                .sorted()
                .toList();

        Map<String, List<String>> genreMap = new ConcurrentHashMap<>();
        for (Rating r : ratings) {
            genreMap.putIfAbsent(r.getTitle(), r.getGenres());
        }

        ConcurrentSkipListMap<String, Map<String, Object>> output = new ConcurrentSkipListMap<>();

        ExecutorService executor = Executors.newFixedThreadPool(
                Math.max(2, Runtime.getRuntime().availableProcessors()), Thread.ofPlatform().factory());

        List<Callable<Void>> tasks = new ArrayList<>();

        for (String user : allUsersOrdered) {
            tasks.add(() -> {
                Map<String, Object> userEntry = new LinkedHashMap<>();
                userEntry.put("user_id", user);

                List<Map<String, Object>> movieList = new ArrayList<>();
                Map<String, Double> userRatings = predictedMatrix.getOrDefault(user, Map.of());

                for (String title : allTitlesOrdered) {
                    Map<String, Object> movieData = new LinkedHashMap<>();
                    movieData.put("title", title);
                    movieData.put("genre", genreMap.getOrDefault(title, List.of()));
                    Double rating = userRatings.get(title);
                    movieData.put("rating", rating != null ? rating : "null");
                    movieList.add(movieData);
                }

                userEntry.put("movies", movieList);
                output.put(user, userEntry);
                return null;
            });
        }

        executor.invokeAll(tasks);
        executor.shutdown();

        try (FileWriter writer = new FileWriter(filename)) {
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            gson.toJson(output.values(), writer);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    private static <T> List<List<T>> splitList(List<T> list, int chunkSize) {
        List<List<T>> chunks = new ArrayList<>();
        for (int i = 0; i < list.size(); i += chunkSize) {
            chunks.add(list.subList(i, Math.min(i + chunkSize, list.size())));
        }
        return chunks;
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