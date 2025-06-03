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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.google.gson.*;
import com.google.gson.reflect.TypeToken;
import com.google.gson.stream.JsonWriter;
import org.jetbrains.annotations.NotNull;


public class MovieRecommenderSGD {

    private static class Rating {
        String user_id;
        String title;
        // O nome do campo foi alterado de 'genre' para 'genres'
        // para corresponder ao JSON de entrada.
        List<String> genres; // <<--- MODIFICADO AQUI
        double rating;

        public Rating(String user_id, String title, List<String> genres, double rating) { // <<--- MODIFICADO AQUI (parâmetro)
            this.user_id = user_id;
            this.title = title;
            this.genres = genres; // <<--- MODIFICADO AQUI (atribuição)
            this.rating = rating;
        }

        public String getUserId() { return user_id; }
        public String getTitle() { return title; }

        // O getter já estava nomeado corretamente como getGenres(),
        // agora ele retorna o campo 'genres' correto.
        public List<String> getGenres() { return genres; } // <<--- Retorna o campo 'genres'
        public double getRating() { return rating; }
    }

    private static final Gson gson = new Gson();
    private static final Type ratingListType = new TypeToken<List<Rating>>() {}.getType();

    private static final AtomicInteger predicts = new AtomicInteger();

    private static final int NUM_FEATURES = 10;
    private static final double LEARNING_RATE = 0.01;
    private static final double REGULARIZATION = 0.02;
    private static final int NUM_EPOCHS = 100;

    private static final ConcurrentHashMap<String, double[]> userFactors = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, double[]> genreFactors = new ConcurrentHashMap<>();

    private static final ConcurrentHashMap<String, ReentrantLock> userLocks = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<String, ReentrantLock> genreLocks = new ConcurrentHashMap<>();


    private static final Set<String> allGenres = ConcurrentHashMap.newKeySet();
    private static final Set<String> allUsers = ConcurrentHashMap.newKeySet();



    private static ConcurrentLinkedQueue<Rating> loadRatingsParallel(Set<String> filenames) {
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
    private static ConcurrentLinkedQueue<Rating> loadRatingsParallel(String filename) {
        return loadRatingsParallel(Set.of(filename));
    }


    public static void initializeFactors(@NotNull ConcurrentLinkedQueue<Rating> ratings) throws InterruptedException {
        allUsers.clear();
        allGenres.clear();

        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        for (Rating r : ratings) {
            executor.submit(() -> {
                allUsers.add(r.getUserId());
                if (r.getGenres() != null) {
                    allGenres.addAll(r.getGenres());
                }
            });
        }
        executor.shutdown();

        ExecutorService factorInitExecutor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        for (String user : allUsers) {
            factorInitExecutor.submit(() -> {
                double[] features = ThreadLocalRandom.current().doubles(NUM_FEATURES, 0, 0.1).toArray();
                userFactors.put(user, features);
                userLocks.put(user, new ReentrantLock());
            });
        }
        for (String genre : allGenres) {
            factorInitExecutor.submit(() -> {
                double[] features = ThreadLocalRandom.current().doubles(NUM_FEATURES, 0, 0.1).toArray();
                genreFactors.put(genre, features);
                genreLocks.put(genre, new ReentrantLock());
            });
        }
        factorInitExecutor.shutdown();
    }

    private static void trainModel(ConcurrentLinkedQueue<Rating> ratings) {
        System.out.println("Iniciando Treinamento");
        List<Rating> ratingList = new ArrayList<>(ratings);

        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            ratingList.parallelStream().forEach(rating -> {
                String user = rating.getUserId();
                double[] userVec = userFactors.get(user);
                ReentrantLock uLock = userLocks.get(user);

                if (userVec == null || uLock == null) {
                    return;
                }

                List<String> currentGenres = rating.getGenres();
                if (currentGenres == null) return;

                for (String genre : currentGenres) {
                    double[] genreVec = genreFactors.get(genre);
                    ReentrantLock gLock = genreLocks.get(genre);

                    if (genreVec == null || gLock == null) {
                          continue;
                    }

                    ReentrantLock firstLock;
                    ReentrantLock secondLock;

                    if (user.compareTo(genre) < 0) {
                        firstLock = uLock;
                        secondLock = gLock;
                    } else if (user.compareTo(genre) > 0) {
                        firstLock = gLock;
                        secondLock = uLock;
                    } else {
                        firstLock = uLock;
                        secondLock = gLock;
                    }

                    firstLock.lock();
                    try {
                        secondLock.lock();
                        try {
                            double prediction = dot(userVec, genreVec);
                            double error = rating.getRating() - prediction;
                            updateVectors(userVec, genreVec, error);
                        } finally {
                            secondLock.unlock();
                        }
                    } finally {
                        firstLock.unlock();
                    }
                }
            });
            if ((epoch + 1) % 10 == 0) {
                System.out.println("Epoch " + (epoch + 1) + "/" + NUM_EPOCHS + " concluída.");
            }
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
        ConcurrentMap<String, List<Rating>> ratingsByUser = new ConcurrentHashMap<>();
        for (Rating r : ratings) {
            ratingsByUser.computeIfAbsent(r.getUserId(), k -> new ArrayList<>()).add(r);
        }
        ConcurrentMap<String, List<Rating>> ratingsByTitle = new ConcurrentHashMap<>();
        for (Rating r : ratings) {
            ratingsByTitle.computeIfAbsent(r.getTitle(), k -> new ArrayList<>()).add(r);
        }
        ConcurrentMap<String, Map<String, Double>> matrix = new ConcurrentHashMap<>();
        int numThreads = Math.max(2, Runtime.getRuntime().availableProcessors());
        ExecutorService executor = Executors.newFixedThreadPool(numThreads, Thread.ofPlatform().factory());
        List<String> userList = new ArrayList<>(allUsers);
        if (userList.isEmpty() && !ratings.isEmpty()) {
            ratings.stream().map(Rating::getUserId).distinct().forEach(userList::add);
        }
        int chunkSize = (userList.isEmpty()) ? 0 : (int) Math.ceil(userList.size() / (double) numThreads);
        if (chunkSize == 0 && !userList.isEmpty()) chunkSize = 1;

        List<Callable<Void>> tasks = new ArrayList<>();
        for (int i = 0; i < userList.size(); i += chunkSize) {
            int end = Math.min(i + chunkSize, userList.size());
            List<String> chunk = userList.subList(i, end);
            tasks.add(() -> {
                for (String user : chunk) {
                    double[] userVec = userFactors.get(user);
                    if (userVec == null) continue;
                    Map<String, Double> userRatings = new HashMap<>();
                    List<Rating> knownRatings = ratingsByUser.get(user);
                    if (knownRatings != null) {
                        for (Rating r : knownRatings) {
                            userRatings.put(r.getTitle(), r.getRating());
                            predicts.getAndIncrement();
                        }
                    }
                    for (Map.Entry<String, List<Rating>> entry : ratingsByTitle.entrySet()) {
                        String title = entry.getKey();
                        if (userRatings.containsKey(title)) continue;
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
                    matrix.put(user, userRatings);
                    //predicts.getAndIncrement();
                }
                return null;
            });
        }
        if (!tasks.isEmpty()) {
            executor.invokeAll(tasks);
        }
        executor.shutdown();

        System.out.println("\n--------------------------------------------------");
        System.out.println("Total de previsões (predicts) realizadas: " + predicts);
        System.out.println("--------------------------------------------------\n");

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


    public static double dot(double[] a, double[] b) {
        if (a == null || b == null || a.length != b.length) return 0.0;
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



    public static Set<String> carregarArquivosDaPasta(String caminhoDaPasta) throws IOException {
        System.out.println("Lendo arquivos da pasta: " + caminhoDaPasta);
        try (Stream<Path> streamDePaths = Files.list(Paths.get(caminhoDaPasta))) {
            Set<String> arquivosEncontrados = streamDePaths
                    .filter(Files::isRegularFile)
                    .peek(path -> System.out.println("Encontrado arquivo: " + path.toString()))
                    .map(Path::toString)
                    .collect(Collectors.toSet());
            if (arquivosEncontrados.isEmpty()) {
                System.out.println("Nenhum arquivo encontrado na pasta: " + caminhoDaPasta);
            }
            return arquivosEncontrados;
        }
    }



    public static void main(String[] args) throws Exception {

        /*
        Set<String> arquivos;
        String pastaDeDatasets = "dataset/avaliacao_individual";
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

        Set<String> arquivos = Set.of("dataset/avaliacoes_completas1GB.json");



        long startTime = System.nanoTime();

        ConcurrentLinkedQueue<Rating> ratings = loadRatingsParallel(arquivos);

        long readTime = System.nanoTime();
        System.out.printf("lidos em: %.2f segundos%n", (readTime - startTime) / 1e9);

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