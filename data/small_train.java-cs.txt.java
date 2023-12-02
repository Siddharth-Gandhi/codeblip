public ListSpeechSynthesisTasksResult listSpeechSynthesisTasks(ListSpeechSynthesisTasksRequest request) {request = beforeClientExecution(request);return executeListSpeechSynthesisTasks(request);}
public UpdateJourneyStateResult updateJourneyState(UpdateJourneyStateRequest request) {request = beforeClientExecution(request);return executeUpdateJourneyState(request);}
public void removePresentationFormat() {remove1stProperty(PropertyIDMap.PID_PRESFORMAT);}
public CellRangeAddressList(int firstRow, int lastRow, int firstCol, int lastCol) {this();addCellRangeAddress(firstRow, firstCol, lastRow, lastCol);}
public void delete(int key) {int i = binarySearch(mKeys, 0, mSize, key);if (i >= 0) {if (mValues[i] != DELETED) {mValues[i] = DELETED;mGarbage = true;}}}
public CreateBranchCommand setStartPoint(RevCommit startPoint) {checkCallable();this.startCommit = startPoint;this.startPoint = null;return this;}
public int centerX() {return x + w / 2;}
public ListPresetsResult listPresets() {return listPresets(new ListPresetsRequest());}
public DeleteFolderContentsResult deleteFolderContents(DeleteFolderContentsRequest request) {request = beforeClientExecution(request);return executeDeleteFolderContents(request);}
public GetConsoleOutputResult getConsoleOutput(GetConsoleOutputRequest request) {request = beforeClientExecution(request);return executeGetConsoleOutput(request);}
public PutMailboxPermissionsResult putMailboxPermissions(PutMailboxPermissionsRequest request) {request = beforeClientExecution(request);return executePutMailboxPermissions(request);}
public Cluster disableSnapshotCopy(DisableSnapshotCopyRequest request) {request = beforeClientExecution(request);return executeDisableSnapshotCopy(request);}
public static String stripExtension(String filename) {int idx = filename.indexOf('.');if (idx != -1) {filename = filename.substring(0, idx);}return filename;}
public ByteBuffer putInt(int value) {throw new ReadOnlyBufferException();}
public int lastIndexOf(final int o){int rval = _limit - 1;for (; rval >= 0; rval--){if (o == _array[ rval ]){break;}}return rval;}
public void setCountsByTime(int[] counts, long msecStep) {countsByTime = counts;countsByTimeStepMSec = msecStep;}
public FeatHdrRecord(RecordInputStream in) {futureHeader = new FtrHeader(in);isf_sharedFeatureType = in.readShort();reserved = in.readByte();cbHdrData = in.readInt();rgbHdrData = in.readRemainder();}
public CopyOnWriteArrayList() {elements = EmptyArray.OBJECT;}
public WriteRequest(DeleteRequest deleteRequest) {setDeleteRequest(deleteRequest);}
public void readFully(byte[] buf){_in.readFully(buf);}
public static Cell getCell(Row row, int columnIndex) {Cell cell = row.getCell(columnIndex);if (cell == null) {cell = row.createCell(columnIndex);}return cell;}
public void setPackConfig(PackConfig pc) {this.packConfig = pc;}
public String getSignerName() {return "HMAC-SHA1";}
public IntervalSet or(IntSet a) {IntervalSet o = new IntervalSet();o.addAll(this);o.addAll(a);return o;}
public String toString() {return getClass().getName() + " [" +_value +"]";}
public DescribeVpcEndpointServicePermissionsResult describeVpcEndpointServicePermissions(DescribeVpcEndpointServicePermissionsRequest request) {request = beforeClientExecution(request);return executeDescribeVpcEndpointServicePermissions(request);}
public static byte[] copyOfRange(byte[] original, int start, int end) {if (start > end) {throw new IllegalArgumentException();}int originalLength = original.length;if (start < 0 || start > originalLength) {throw new ArrayIndexOutOfBoundsException();}int resultLength = end - start;int copyLength = Math.min(resultLength, originalLength - start);byte[] result = new byte[resultLength];System.arraycopy(original, start, result, 0, copyLength);return result;}
public ListTopicsRequest(String nextToken) {setNextToken(nextToken);}
public void finish(FieldInfos fis, int numDocs) throws IOException {if (!pendingDocs.isEmpty()) {flush();numDirtyChunks++; }if (numDocs != this.numDocs) {throw new RuntimeException("Wrote " + this.numDocs + " docs, finish called with numDocs=" + numDocs);}indexWriter.finish(numDocs, vectorsStream.getFilePointer());vectorsStream.writeVLong(numChunks);vectorsStream.writeVLong(numDirtyChunks);CodecUtil.writeFooter(vectorsStream);}
public boolean isIndexTerm(BytesRef term, TermStats stats) {if (count >= interval) {count = 1;return true;} else {count++;return false;}}
public AssociateDhcpOptionsResult associateDhcpOptions(AssociateDhcpOptionsRequest request) {request = beforeClientExecution(request);return executeAssociateDhcpOptions(request);}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval arg0, ValueEval arg1,ValueEval arg2) {return evaluate(srcRowIndex, srcColumnIndex, arg0, arg1, arg2, DEFAULT_ARG3);}
public void disconnect() {if (sock.isConnected())sock.disconnect();}
public PredictionContext add(PredictionContext ctx) {if ( ctx==PredictionContext.EMPTY ) return PredictionContext.EMPTY;PredictionContext existing = cache.get(ctx);if ( existing!=null ) {return existing;}cache.put(ctx, ctx);return ctx;}
public UploadLayerPartResult uploadLayerPart(UploadLayerPartRequest request) {request = beforeClientExecution(request);return executeUploadLayerPart(request);}
public String getScriptText() {return getScriptText(null, null);}
public DescribeClusterSubnetGroupsResult describeClusterSubnetGroups() {return describeClusterSubnetGroups(new DescribeClusterSubnetGroupsRequest());}
public char setIndex(int position) {if (position < getBeginIndex() || position > getEndIndex())throw new IllegalArgumentException("Illegal Position: " + position);index = start + position;return current();}
public GetPhoneNumberOrderResult getPhoneNumberOrder(GetPhoneNumberOrderRequest request) {request = beforeClientExecution(request);return executeGetPhoneNumberOrder(request);}
public EpsilonTransition(ATNState target, int outermostPrecedenceReturn) {super(target);this.outermostPrecedenceReturn = outermostPrecedenceReturn;}
public DiffCommand setContextLines(int contextLines) {this.contextLines = contextLines;return this;}
public RejectVpcPeeringConnectionResult rejectVpcPeeringConnection(RejectVpcPeeringConnectionRequest request) {request = beforeClientExecution(request);return executeRejectVpcPeeringConnection(request);}
public static boolean equals(int[] array1, int[] array2) {if (array1 == array2) {return true;}if (array1 == null || array2 == null || array1.length != array2.length) {return false;}for (int i = 0; i < array1.length; i++) {if (array1[i] != array2[i]) {return false;}}return true;}
public static void main(String[] args) throws IOException {if (args.length<1) {System.err.println("Usage: java QualityQueriesFinder <index-dir>");System.exit(1);}QualityQueriesFinder qqf = new QualityQueriesFinder(FSDirectory.open(Paths.get(args[0])));String q[] = qqf.bestQueries("body",20);for (int i=0; i<q.length; i++) {System.out.println(newline+formatQueryAsTrecTopic(i,q[i],null,null));}}
public CharsRef(char[] chars, int offset, int length) {this.chars = chars;this.offset = offset;this.length = length;assert isValid();}
public UpdateIPSetResult updateIPSet(UpdateIPSetRequest request) {request = beforeClientExecution(request);return executeUpdateIPSet(request);}
public void print(Object obj) {print(String.valueOf(obj));}
public String toString() {return "IndexFileDeleter.CommitPoint(" + segmentsFileName + ")";}
public synchronized boolean waitForGeneration(long targetGen, int maxMS) throws InterruptedException {if (targetGen > searchingGen) {reopenLock.lock();waitingGen = Math.max(waitingGen, targetGen);try {reopenCond.signal();} finally {reopenLock.unlock();}long startMS = System.nanoTime()/1000000;while (targetGen > searchingGen) {if (maxMS < 0) {wait();} else {long msLeft = (startMS + maxMS) - System.nanoTime()/1000000;if (msLeft <= 0) {return false;} else {wait(msLeft);}}}}return true;}
public StringBuffer append(boolean b) {return append(b ? "true" : "false");}
public ByteBuffer put(int index, byte b) {throw new ReadOnlyBufferException();}
public int getLineCount() {return lineCount;}
public boolean equals( Object o ) {return o instanceof DutchStemmer;}
public CreateNotificationSubscriptionResult createNotificationSubscription(CreateNotificationSubscriptionRequest request) {request = beforeClientExecution(request);return executeCreateNotificationSubscription(request);}
public boolean isOutdated() {return snapshot.isModified(getFile());}
public DescribeVirtualInterfacesResult describeVirtualInterfaces() {return describeVirtualInterfaces(new DescribeVirtualInterfacesRequest());}
public void onChanged() {buildMap();for (DataSetObserver o : mObservers) {o.onChanged();}}
public DeleteEventTrackerResult deleteEventTracker(DeleteEventTrackerRequest request) {request = beforeClientExecution(request);return executeDeleteEventTracker(request);}
public boolean matches(ValueEval x) {if (x instanceof BlankEval) {switch(getCode()) {case CmpOp.NONE:case CmpOp.EQ:return _value.length() == 0;case CmpOp.NE:return _value.length() != 0;}return false;}if(!(x instanceof StringEval)) {return false;}String testedValue = ((StringEval) x).getStringValue();if (testedValue.length() < 1 && _value.length() < 1) {switch(getCode()) {case CmpOp.NONE: return true;case CmpOp.EQ:   return false;case CmpOp.NE:   return true;}return false;}if (_pattern != null) {return evaluate(_pattern.matcher(testedValue).matches());}return evaluate(testedValue.compareToIgnoreCase(_value));}
public ListWebsiteAuthorizationProvidersResult listWebsiteAuthorizationProviders(ListWebsiteAuthorizationProvidersRequest request) {request = beforeClientExecution(request);return executeListWebsiteAuthorizationProviders(request);}
public void write(char[] buf, int offset, int count) {doWrite(buf, offset, count);}
public String formatAsString() {if(isWholeColumnReference()) {returnCellReference.convertNumToColString(_firstCell.getCol())+ ":" +CellReference.convertNumToColString(_lastCell.getCol());}StringBuilder sb = new StringBuilder(32);sb.append(_firstCell.formatAsString());if(!_isSingleCell) {sb.append(CELL_DELIMITER);if(_lastCell.getSheetName() == null) {sb.append(_lastCell.formatAsString());} else {_lastCell.appendCellReference(sb);}}return sb.toString();}
public Graphics create(){return new EscherGraphics(escherGroup, workbook,foreground, font, verticalPointsPerPixel );}
public DoubleDocValues(ValueSource vs) {this.vs = vs;}
public static CharArraySet getDefaultStopSet(){return DefaultSetHolder.DEFAULT_STOP_SET;}
public DeleteLoadBalancerPolicyResult deleteLoadBalancerPolicy(DeleteLoadBalancerPolicyRequest request) {request = beforeClientExecution(request);return executeDeleteLoadBalancerPolicy(request);}
public ReplicationGroup decreaseReplicaCount(DecreaseReplicaCountRequest request) {request = beforeClientExecution(request);return executeDecreaseReplicaCount(request);}
public Result update(RevWalk walk) throws IOException {requireCanDoUpdate();try {return result = updateImpl(walk, new Store() {@OverrideResult execute(Result status) throws IOException {if (status == Result.NO_CHANGE)return status;return doUpdate(status);}});} catch (IOException x) {result = Result.IO_FAILURE;throw x;}}
public Set<String> getChanged() {return Collections.unmodifiableSet(diff.getChanged());}
public static String toHex(long value) {StringBuilder sb = new StringBuilder(16);writeHex(sb, value, 16, "");return sb.toString();}
public int createPlaceholder() {return _offset++;}
@Override public boolean equals(Object o) {if (o instanceof Map.Entry) {Map.Entry other = (Map.Entry) o;return (key == null ? other.getKey() == null : key.equals(other.getKey()))&& (value == null ? other.getValue() == null : value.equals(other.getValue()));}return false;}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval arg0,ValueEval arg1) {double result;try {double d0 = NumericFunction.singleOperandEvaluate(arg0, srcRowIndex, srcColumnIndex);double d1 = NumericFunction.singleOperandEvaluate(arg1, srcRowIndex, srcColumnIndex);double logE = Math.log(d0);if (Double.compare(d1, Math.E) == 0) {result = logE;} else {result = logE / Math.log(d1);}NumericFunction.checkValue(result);} catch (EvaluationException e) {return e.getErrorEval();}return new NumberEval(result);}
public DeleteFilterResult deleteFilter(DeleteFilterRequest request) {request = beforeClientExecution(request);return executeDeleteFilter(request);}
public CreateInstanceSnapshotResult createInstanceSnapshot(CreateInstanceSnapshotRequest request) {request = beforeClientExecution(request);return executeCreateInstanceSnapshot(request);}
public List<Token> getTokens(int start, int stop) {return getTokens(start, stop, null);}
public static TermGroupFacetCollector createTermGroupFacetCollector(String groupField,String facetField,boolean facetFieldMultivalued,BytesRef facetPrefix,int initialSize) {if (facetFieldMultivalued) {return new MV(groupField, facetField, facetPrefix, initialSize);} else {return new SV(groupField, facetField, facetPrefix, initialSize);}}
public RenameAlbumRequest() {super("CloudPhoto", "2017-07-11", "RenameAlbum", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
@Override public boolean contains(Object object) {synchronized (mutex) {return c.contains(object);}}
public CharBuffer put(char[] src, int srcOffset, int charCount) {if (charCount > remaining()) {throw new BufferOverflowException();}System.arraycopy(src, srcOffset, backingArray, offset + position, charCount);position += charCount;return this;}
public LegendRecord(RecordInputStream in) {field_1_xAxisUpperLeft = in.readInt();field_2_yAxisUpperLeft = in.readInt();field_3_xSize          = in.readInt();field_4_ySize          = in.readInt();field_5_type           = in.readByte();field_6_spacing        = in.readByte();field_7_options        = in.readShort();}
public static byte[] encodedTypeString(int typeCode) {switch (typeCode) {case OBJ_COMMIT:return ENCODED_TYPE_COMMIT;case OBJ_TREE:return ENCODED_TYPE_TREE;case OBJ_BLOB:return ENCODED_TYPE_BLOB;case OBJ_TAG:return ENCODED_TYPE_TAG;default:throw new IllegalArgumentException(MessageFormat.format(JGitText.get().badObjectType, Integer.valueOf(typeCode)));}}
public ObjectId getCalulatedPatchId() {return ObjectId.fromRaw(digest.digest());}
public DefaultRowHeightRecord() {field_1_option_flags = 0x0000;field_2_row_height = DEFAULT_ROW_HEIGHT;}
public final ByteBuffer encode(CharBuffer buffer) {try {return newEncoder().onMalformedInput(CodingErrorAction.REPLACE).onUnmappableCharacter(CodingErrorAction.REPLACE).encode(buffer);} catch (CharacterCodingException ex) {throw new Error(ex.getMessage(), ex);}}
public final FloatBuffer get(float[] dst, int dstOffset, int floatCount) {if (floatCount > remaining()) {throw new BufferUnderflowException();}System.arraycopy(backingArray, offset + position, dst, dstOffset, floatCount);position += floatCount;return this;}
public boolean hasNext() {return nextEntry != null;}
public DeleteNatGatewayResult deleteNatGateway(DeleteNatGatewayRequest request) {request = beforeClientExecution(request);return executeDeleteNatGateway(request);}
public String resolveNameXText(int refIndex, int definedNameIndex) {return linkTable.resolveNameXText(refIndex, definedNameIndex, this);}
public void setMultiFields(CharSequence[] fields) {if (fields == null) {fields = new CharSequence[0];}getQueryConfigHandler().set(ConfigurationKeys.MULTI_FIELDS, fields);}
public boolean isCancelled() {lock.lock();try {return pm.isCancelled();} finally {lock.unlock();}}
public RemoveTargetsResult removeTargets(RemoveTargetsRequest request) {request = beforeClientExecution(request);return executeRemoveTargets(request);}
public FuzzyQuery(Term term, int maxEdits, int prefixLength, int maxExpansions, boolean transpositions) {super(term.field());if (maxEdits < 0 || maxEdits > LevenshteinAutomata.MAXIMUM_SUPPORTED_DISTANCE) {throw new IllegalArgumentException("maxEdits must be between 0 and " + LevenshteinAutomata.MAXIMUM_SUPPORTED_DISTANCE);}if (prefixLength < 0) {throw new IllegalArgumentException("prefixLength cannot be negative.");}if (maxExpansions <= 0) {throw new IllegalArgumentException("maxExpansions must be positive.");}this.term = term;this.maxEdits = maxEdits;this.prefixLength = prefixLength;this.transpositions = transpositions;this.maxExpansions = maxExpansions;int[] codePoints = FuzzyTermsEnum.stringToUTF32(term.text());this.termLength = codePoints.length;this.automata = FuzzyTermsEnum.buildAutomata(term.text(), codePoints, prefixLength, transpositions, maxEdits);setRewriteMethod(new MultiTermQuery.TopTermsBlendedFreqScoringRewrite(maxExpansions));this.ramBytesUsed = calculateRamBytesUsed(term, this.automata);}
public CheckoutCommand checkout() {return new CheckoutCommand(repo);}
public ValueEval evaluate(String sheetName, int rowIndex, int columnIndex) {EvaluationCell cell = _sewb.getEvaluationCell(sheetName, rowIndex, columnIndex);switch (cell.getCellType()) {case BOOLEAN:return BoolEval.valueOf(cell.getBooleanCellValue());case ERROR:return ErrorEval.valueOf(cell.getErrorCellValue());case FORMULA:return _evaluator.evaluate(cell);case NUMERIC:return new NumberEval(cell.getNumericCellValue());case STRING:return new StringEval(cell.getStringCellValue());case BLANK:return null;default:throw new IllegalStateException("Bad cell type (" + cell.getCellType() + ")");}}
public PutFileSystemPolicyResult putFileSystemPolicy(PutFileSystemPolicyRequest request) {request = beforeClientExecution(request);return executePutFileSystemPolicy(request);}
public long fileLength(String name) throws IOException {ensureOpen();FileEntry e = entries.get(IndexFileNames.stripSegmentName(name));if (e == null)throw new FileNotFoundException(name);return e.length;}
public DescribeCacheClustersResult describeCacheClusters() {return describeCacheClusters(new DescribeCacheClustersRequest());}
public void setObjectId(RevObject obj) {setObjectId(obj, obj.getType());}
public boolean rowHasCells(int row) {if (row >= records.length) {return false;}CellValueRecordInterface[] rowCells=records[row];if(rowCells==null) return false;for(int col=0;col<rowCells.length;col++) {if(rowCells[col]!=null) return true;}return false;}
public TokenStream create(TokenStream input) {return new SpanishLightStemFilter(input);}
public StoredField(String name, double value) {super(name, TYPE);fieldsData = value;}
public DescribePublicIpv4PoolsResult describePublicIpv4Pools(DescribePublicIpv4PoolsRequest request) {request = beforeClientExecution(request);return executeDescribePublicIpv4Pools(request);}
public IndexRevision(IndexWriter writer) throws IOException {IndexDeletionPolicy delPolicy = writer.getConfig().getIndexDeletionPolicy();if (!(delPolicy instanceof SnapshotDeletionPolicy)) {throw new IllegalArgumentException("IndexWriter must be created with SnapshotDeletionPolicy");}this.writer = writer;this.sdp = (SnapshotDeletionPolicy) delPolicy;this.commit = sdp.snapshot();this.version = revisionVersion(commit);this.sourceFiles = revisionFiles(commit);}
public void setTabIdArray(short[] array) {_tabids = array.clone();}
public UpdateObjectAttributesResult updateObjectAttributes(UpdateObjectAttributesRequest request) {request = beforeClientExecution(request);return executeUpdateObjectAttributes(request);}
public GetGameSessionLogUrlResult getGameSessionLogUrl(GetGameSessionLogUrlRequest request) {request = beforeClientExecution(request);return executeGetGameSessionLogUrl(request);}
public RefCount(T object) {this.object = object;}
public ByteBuffer put(int index, byte b) {checkIndex(index);backingArray[offset + index] = b;return this;}
public IntervalSet LOOK(ATNState s, ATNState stopState, RuleContext ctx) {IntervalSet r = new IntervalSet();boolean seeThruPreds = true; PredictionContext lookContext = ctx != null ? PredictionContext.fromRuleContext(s.atn, ctx) : null;_LOOK(s, stopState, lookContext,r, new HashSet<ATNConfig>(), new BitSet(), seeThruPreds, true);return r;}
public int getValidationType() {return _validationType;}
public DeleteTagCommand tagDelete() {return new DeleteTagCommand(repo);}
public SortRescorer(Sort sort) {this.sort = sort;}
public void verifyBelongsToWorkbook(HSSFWorkbook wb) {if(wb.getWorkbook() != _workbook) {throw new IllegalArgumentException("This Style does not belong to the supplied Workbook. Are you trying to assign a style from one workbook to the cell of a differnt workbook?");}}
public StringBuffer insert(int index, Object obj) {return insert(index, obj == null ? "null" : obj.toString());}
public boolean containsKey(CharSequence cs) {if(cs == null)throw new NullPointerException();return false;}
public int compareTo(HSSFRichTextString r) {return _string.compareTo(r._string);}
public RequestSpotInstancesRequest(String spotPrice) {setSpotPrice(spotPrice);}
public ObjectId getNewObjectId() {return newObjectId;}
public void setDeltaBaseAsOffset(boolean deltaBaseAsOffset) {this.deltaBaseAsOffset = deltaBaseAsOffset;}
public LengthFilterFactory(Map<String, String> args) {super(args);min = requireInt(args, MIN_KEY);max = requireInt(args, MAX_KEY);if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public TruncateTokenFilter(TokenStream input, int length) {super(input);if (length < 1)throw new IllegalArgumentException("length parameter must be a positive number: " + length);this.length = length;}
public ListDomainsResult listDomains() {return listDomains(new ListDomainsRequest());}
public ArabicStemFilter create(TokenStream input) {return new ArabicStemFilter(input);}
public PushCommand setRefSpecs(RefSpec... specs) {checkCallable();this.refSpecs.clear();Collections.addAll(refSpecs, specs);return this;}
public BlameGenerator setDiffAlgorithm(DiffAlgorithm algorithm) {diffAlgorithm = algorithm;return this;}
public GroupingSearch setIncludeMaxScore(boolean includeMaxScore) {this.includeMaxScore = includeMaxScore;return this;}
public Field[] createIndexableFields(Shape shape) {double distErr = SpatialArgs.calcDistanceFromErrPct(shape, distErrPct, ctx);return createIndexableFields(shape, distErr);}
public PutMethodResponseResult putMethodResponse(PutMethodResponseRequest request) {request = beforeClientExecution(request);return executePutMethodResponse(request);}
public LegacyCredentials(Credential legacyCrendential) {this.legacyCredential = legacyCrendential;}
public DescribeFeatureTransformationResult describeFeatureTransformation(DescribeFeatureTransformationRequest request) {request = beforeClientExecution(request);return executeDescribeFeatureTransformation(request);}
public DeleteRouteResult deleteRoute(DeleteRouteRequest request) {request = beforeClientExecution(request);return executeDeleteRoute(request);}
public AssociatePhoneNumbersWithVoiceConnectorResult associatePhoneNumbersWithVoiceConnector(AssociatePhoneNumbersWithVoiceConnectorRequest request) {request = beforeClientExecution(request);return executeAssociatePhoneNumbersWithVoiceConnector(request);}
public long ramBytesUsed() {long size = BASE_RAM_BYTES_USED + RamUsageEstimator.shallowSizeOf(blocks);if (blocks.length > 0) {size += (blocks.length - 1) * bytesUsedPerBlock;size += RamUsageEstimator.sizeOf(blocks[blocks.length - 1]);}return size;}
public short readShort(){return _in.readShort();}
public UpdatePipelineNotificationsResult updatePipelineNotifications(UpdatePipelineNotificationsRequest request) {request = beforeClientExecution(request);return executeUpdatePipelineNotifications(request);}
public StringWriter append(char c) {write(c);return this;}
public Iterator<V> iterator() {return new ValueIterator();}
public UnitsRecord(RecordInputStream in) {field_1_units = in.readShort();}
public boolean isEmpty() {return first;}
public String toString() {return "ANY_DIFF"; }
public UpdateDomainNameResult updateDomainName(UpdateDomainNameRequest request) {request = beforeClientExecution(request);return executeUpdateDomainName(request);}
public DeleteSnapshotRequest(String snapshotId) {setSnapshotId(snapshotId);}
public void readFully(byte[] buf) {readFully(buf, 0, buf.length);}
public SliceReader(IntBlockPool pool) {this.pool = pool;}
public void setDeltaSearchMemoryLimit(long memoryLimit) {deltaSearchMemoryLimit = memoryLimit;}
public String toString(){StringBuilder buffer = new StringBuilder();buffer.append("[BOOKBOOL]\n");buffer.append("    .savelinkvalues  = ").append(Integer.toHexString(getSaveLinkValues())).append("\n");buffer.append("[/BOOKBOOL]\n");return buffer.toString();}
public DescribeTransitGatewayAttachmentsResult describeTransitGatewayAttachments(DescribeTransitGatewayAttachmentsRequest request) {request = beforeClientExecution(request);return executeDescribeTransitGatewayAttachments(request);}
public CreateVpcResult createVpc(CreateVpcRequest request) {request = beforeClientExecution(request);return executeCreateVpc(request);}
public DescribeElasticGpusResult describeElasticGpus(DescribeElasticGpusRequest request) {request = beforeClientExecution(request);return executeDescribeElasticGpus(request);}
public IntBuffer put(int c) {if (position == limit) {throw new BufferOverflowException();}byteBuffer.putInt(position++ * SizeOf.INT, c);return this;}
public UpdateEndpointsBatchResult updateEndpointsBatch(UpdateEndpointsBatchRequest request) {request = beforeClientExecution(request);return executeUpdateEndpointsBatch(request);}
public void fromRaw(byte[] bs, int p) {w1 = NB.decodeInt32(bs, p);w2 = NB.decodeInt32(bs, p + 4);w3 = NB.decodeInt32(bs, p + 8);w4 = NB.decodeInt32(bs, p + 12);w5 = NB.decodeInt32(bs, p + 16);}
public static OpenSshConfig get(FS fs) {File home = fs.userHome();if (home == null)home = new File(".").getAbsoluteFile(); final File config = new File(new File(home, SshConstants.SSH_DIR),SshConstants.CONFIG);return new OpenSshConfig(home, config);}
public VCenterRecord(RecordInputStream in) {field_1_vcenter = in.readShort();}
public synchronized InputStream obtainFile(String sessionID, String source, String fileName) throws IOException {ensureOpen();ReplicationSession session = sessions.get(sessionID);if (session != null && session.isExpired(expirationThresholdMilllis)) {releaseSession(sessionID);session = null;}if (session == null) {throw new SessionExpiredException("session (" + sessionID + ") expired while obtaining file: source=" + source+ " file=" + fileName);}sessions.get(sessionID).markAccessed();return session.revision.revision.open(source, fileName);}
public DownloadDefaultKeyPairResult downloadDefaultKeyPair(DownloadDefaultKeyPairRequest request) {request = beforeClientExecution(request);return executeDownloadDefaultKeyPair(request);}
public DescribeLocalGatewayRouteTableVirtualInterfaceGroupAssociationsResult describeLocalGatewayRouteTableVirtualInterfaceGroupAssociations(DescribeLocalGatewayRouteTableVirtualInterfaceGroupAssociationsRequest request) {request = beforeClientExecution(request);return executeDescribeLocalGatewayRouteTableVirtualInterfaceGroupAssociations(request);}
public ResetEbsDefaultKmsKeyIdResult resetEbsDefaultKmsKeyId(ResetEbsDefaultKmsKeyIdRequest request) {request = beforeClientExecution(request);return executeResetEbsDefaultKmsKeyId(request);}
public int getPropertiesPerBlock() {return bigBlockSize / POIFSConstants.PROPERTY_SIZE;}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval numberVE) {return this.evaluate(srcRowIndex, srcColumnIndex, numberVE, null);}
public GetFindingsStatisticsResult getFindingsStatistics(GetFindingsStatisticsRequest request) {request = beforeClientExecution(request);return executeGetFindingsStatistics(request);}
public DBCluster modifyDBCluster(ModifyDBClusterRequest request) {request = beforeClientExecution(request);return executeModifyDBCluster(request);}
public LimitTokenCountFilterFactory(Map<String, String> args) {super(args);maxTokenCount = requireInt(args, MAX_TOKEN_COUNT_KEY);consumeAllTokens = getBoolean(args, CONSUME_ALL_TOKENS_KEY, false);if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public MatchNoDocsQuery build(QueryNode queryNode) throws QueryNodeException {if (!(queryNode instanceof MatchNoDocsQueryNode)) {throw new QueryNodeException(new MessageImpl(QueryParserMessages.LUCENE_QUERY_CONVERSION_ERROR, queryNode.toQueryString(new EscapeQuerySyntaxImpl()), queryNode.getClass().getName()));}return new MatchNoDocsQuery();}
public GetUserPolicyRequest(String userName, String policyName) {setUserName(userName);setPolicyName(policyName);}
public Cluster rotateEncryptionKey(RotateEncryptionKeyRequest request) {request = beforeClientExecution(request);return executeRotateEncryptionKey(request);}
public int getLinesAdded() {return nAdded;}
public List<Token> getHiddenTokensToLeft(int tokenIndex, int channel) {lazyInit();if ( tokenIndex<0 || tokenIndex>=tokens.size() ) {throw new IndexOutOfBoundsException(tokenIndex+" not in 0.."+(tokens.size()-1));}if (tokenIndex == 0) {return null;}int prevOnChannel =previousTokenOnChannel(tokenIndex - 1, Lexer.DEFAULT_TOKEN_CHANNEL);if ( prevOnChannel == tokenIndex - 1 ) return null;int from = prevOnChannel+1;int to = tokenIndex-1;return filterForChannel(from, to, channel);}
public ValidDBInstanceModificationsMessage describeValidDBInstanceModifications(DescribeValidDBInstanceModificationsRequest request) {request = beforeClientExecution(request);return executeDescribeValidDBInstanceModifications(request);}
public final void add(RevFlag flag) {flags |= flag.mask;}
public void clear() {Iterator<E> it = iterator();while (it.hasNext()) {it.next();it.remove();}}
public RegisterImageResult registerImage(RegisterImageRequest request) {request = beforeClientExecution(request);return executeRegisterImage(request);}
public boolean equals(Object other) {return sameClassAs(other) &&term.equals(((TermQuery) other).term);}
public URI(String scheme, String authority, String path, String query,String fragment) throws URISyntaxException {if (scheme != null && path != null && !path.isEmpty() && path.charAt(0) != '/') {throw new URISyntaxException(path, "Relative path");}StringBuilder uri = new StringBuilder();if (scheme != null) {uri.append(scheme);uri.append(':');}if (authority != null) {uri.append("AUTHORITY_ENCODER.appendEncoded(uri, authority);}if (path != null) {PATH_ENCODER.appendEncoded(uri, path);}if (query != null) {uri.append('?');ALL_LEGAL_ENCODER.appendEncoded(uri, query);}if (fragment != null) {uri.append('#');ALL_LEGAL_ENCODER.appendEncoded(uri, fragment);}parseURI(uri.toString(), false);}
public BlameGenerator(Repository repository, String path) {this.repository = repository;this.resultPath = PathFilter.create(path);idBuf = new MutableObjectId();setFollowFileRenames(true);initRevPool(false);remaining = -1;}
public synchronized void writeTo(OutputStream out) throws IOException {out.write(buf, 0, count);}
public DeletableItem(String name, java.util.List<Attribute> attributes) {setName(name);setAttributes(attributes);}
public DescribeGroupResult describeGroup(DescribeGroupRequest request) {request = beforeClientExecution(request);return executeDescribeGroup(request);}
public EnableVpcClassicLinkResult enableVpcClassicLink(EnableVpcClassicLinkRequest request) {request = beforeClientExecution(request);return executeEnableVpcClassicLink(request);}
public DescribeStacksResult describeStacks() {return describeStacks(new DescribeStacksRequest());}
public CharBuffer duplicate() {return copy(this);}
public static double mod(double n, double d) {if (d == 0) {return Double.NaN;}else if (sign(n) == sign(d)) {return n % d;}else {return ((n % d) + d) % d;}}
public static String getLocalizedMessage(String key, Locale locale) {Object message = getResourceBundleObject(key, locale);if (message == null) {return "Message with key:" + key + " and locale: " + locale+ " not found.";}return message.toString();}
public CharSequence toQueryString(EscapeQuerySyntax escapeSyntaxParser) {if (getChild() == null)return "";return getChild().toQueryString(escapeSyntaxParser) + "^"+ getValueString();}
public CharSequence toQueryString(EscapeQuerySyntax escapeSyntaxParser) {if (getChild() == null)return "";return "( " + getChild().toQueryString(escapeSyntaxParser) + " )";}
public GetInvalidationResult getInvalidation(GetInvalidationRequest request) {request = beforeClientExecution(request);return executeGetInvalidation(request);}
public String formatAsString() {return formatAsString(null, false);}
public final int prefixCompare(byte[] bs, int p) {int cmp;cmp = NB.compareUInt32(w1, mask(1, NB.decodeInt32(bs, p)));if (cmp != 0)return cmp;cmp = NB.compareUInt32(w2, mask(2, NB.decodeInt32(bs, p + 4)));if (cmp != 0)return cmp;cmp = NB.compareUInt32(w3, mask(3, NB.decodeInt32(bs, p + 8)));if (cmp != 0)return cmp;cmp = NB.compareUInt32(w4, mask(4, NB.decodeInt32(bs, p + 12)));if (cmp != 0)return cmp;return NB.compareUInt32(w5, mask(5, NB.decodeInt32(bs, p + 16)));}
public AddApplicationInputProcessingConfigurationResult addApplicationInputProcessingConfiguration(AddApplicationInputProcessingConfigurationRequest request) {request = beforeClientExecution(request);return executeAddApplicationInputProcessingConfiguration(request);}
public static TermRangeQuery newStringRange(String field, String lowerTerm, String upperTerm, boolean includeLower, boolean includeUpper) {BytesRef lower = lowerTerm == null ? null : new BytesRef(lowerTerm);BytesRef upper = upperTerm == null ? null : new BytesRef(upperTerm);return new TermRangeQuery(field, lower, upper, includeLower, includeUpper);}
static public double fv(double r, int nper, double pmt, double pv, int type) {return -(pv * Math.pow(1 + r, nper) + pmt * (1+r*type) * (Math.pow(1 + r, nper) - 1) / r);}
public int checkExternSheet(int firstSheetIndex, int lastSheetIndex) {int thisWbIndex = -1; for (int i = 0; i < _externalBookBlocks.length; i++) {SupBookRecord ebr = _externalBookBlocks[i].getExternalBookRecord();if (ebr.isInternalReferences()) {thisWbIndex = i;break;}}if (thisWbIndex < 0) {throw new RuntimeException("Could not find 'internal references' EXTERNALBOOK");}int i = _externSheetRecord.getRefIxForSheet(thisWbIndex, firstSheetIndex, lastSheetIndex);if (i >= 0) {return i;}return _externSheetRecord.addRef(thisWbIndex, firstSheetIndex, lastSheetIndex);}
public DescribeSentimentDetectionJobResult describeSentimentDetectionJob(DescribeSentimentDetectionJobRequest request) {request = beforeClientExecution(request);return executeDescribeSentimentDetectionJob(request);}
public String toString(){StringBuilder buffer = new StringBuilder();buffer.append("[UNITS]\n");buffer.append("    .units                = ").append("0x").append(HexDump.toHex(  getUnits ())).append(" (").append( getUnits() ).append(" )");buffer.append(System.getProperty("line.separator"));buffer.append("[/UNITS]\n");return buffer.toString();}
public NavigableMap<K, V> tailMap(K from, boolean inclusive) {Bound fromBound = inclusive ? INCLUSIVE : EXCLUSIVE;return subMap(from, fromBound, null, NO_BOUND);}
public static int compareTo(Ref o1, Ref o2) {return o1.getName().compareTo(o2.getName());}
public PutEventsConfigurationResult putEventsConfiguration(PutEventsConfigurationRequest request) {request = beforeClientExecution(request);return executePutEventsConfiguration(request);}
public DetachFromIndexResult detachFromIndex(DetachFromIndexRequest request) {request = beforeClientExecution(request);return executeDetachFromIndex(request);}
public RebaseCommand rebase() {return new RebaseCommand(repo);}
public SearchGroup<T> next() {assert iter.hasNext();final SearchGroup<T> group = iter.next();if (group.sortValues == null) {throw new IllegalArgumentException("group.sortValues is null; you must pass fillFields=true to the first pass collector");}return group;}
public UpdateMLModelResult updateMLModel(UpdateMLModelRequest request) {request = beforeClientExecution(request);return executeUpdateMLModel(request);}
public CreateIPSetResult createIPSet(CreateIPSetRequest request) {request = beforeClientExecution(request);return executeCreateIPSet(request);}
public FieldDateResolutionFCListener(QueryConfigHandler config) {this.config = config;}
@Override public boolean containsValue(Object value) {HashMapEntry[] tab = table;int len = tab.length;if (value == null) {for (int i = 0; i < len; i++) {for (HashMapEntry e = tab[i]; e != null; e = e.next) {if (e.value == null) {return true;}}}return entryForNullKey != null && entryForNullKey.value == null;}for (int i = 0; i < len; i++) {for (HashMapEntry e = tab[i]; e != null; e = e.next) {if (value.equals(e.value)) {return true;}}}return entryForNullKey != null && value.equals(entryForNullKey.value);}
public DescribeWorkspaceBundlesResult describeWorkspaceBundles(DescribeWorkspaceBundlesRequest request) {request = beforeClientExecution(request);return executeDescribeWorkspaceBundles(request);}
public PostingsEnum reset(int[] postings) {this.postings = postings;upto = -1;return this;}
public void serialize(LittleEndianOutput out) {out.writeShort(sid); out.writeShort(_reserved0);out.writeInt(_engineId);}
public static CharBuffer allocate(int capacity) {if (capacity < 0) {throw new IllegalArgumentException();}return new ReadWriteCharArrayBuffer(capacity);}
public String toFormulaString(String[] operands) {StringBuilder buffer = new StringBuilder();buffer.append(operands[ 0 ]);buffer.append(">=");buffer.append(operands[ 1 ]);return buffer.toString();}
public DeletePipelineResult deletePipeline(DeletePipelineRequest request) {request = beforeClientExecution(request);return executeDeletePipeline(request);}
public InterfaceHdrRecord(int codePage) {_codepage = codePage;}
public DescribeScalingParametersResult describeScalingParameters(DescribeScalingParametersRequest request) {request = beforeClientExecution(request);return executeDescribeScalingParameters(request);}
public Entry<K, V> higherEntry(K key) {return immutableCopy(findBounded(key, HIGHER));}
public CreateSpotDatafeedSubscriptionRequest(String bucket) {setBucket(bucket);}
public String getLocalizedMessage() {return getLocalizedMessage(Locale.getDefault());}
public UDFFinder getUDFFinder(){return _uBook.getUDFFinder();}
public ExternalName getExternalName(String nameName, String sheetName, int externalWorkbookNumber) {throw new IllegalStateException("XSSF-style external names are not supported for HSSF");}
public OldFormulaRecord(RecordInputStream ris) {super(ris, ris.getSid() == biff2_sid);if (isBiff2()) {field_4_value = ris.readDouble();} else {long valueLongBits  = ris.readLong();specialCachedValue = FormulaSpecialCachedValue.create(valueLongBits);if (specialCachedValue == null) {field_4_value = Double.longBitsToDouble(valueLongBits);}}if (isBiff2()) {field_5_options = (short)ris.readUByte();} else {field_5_options = ris.readShort();}int expression_len = ris.readShort();int nBytesAvailable = ris.available();field_6_parsed_expr = Formula.read(expression_len, ris, nBytesAvailable);}
public int stem(char s[], int len) {assert s.length >= len + 1 : "this stemmer requires an oversized array of at least 1";len = plural.apply(s, len);len = unification.apply(s, len);len = adverb.apply(s, len);int oldlen;do {oldlen = len;len = augmentative.apply(s, len);} while (len != oldlen);oldlen = len;len = noun.apply(s, len);if (len == oldlen) { len = verb.apply(s, len);}len = vowel.apply(s, len);for (int i = 0; i < len; i++)switch(s[i]) {case 'á': s[i] = 'a'; break;case 'é':case 'ê': s[i] = 'e'; break;case 'í': s[i] = 'i'; break;case 'ó': s[i] = 'o'; break;case 'ú': s[i] = 'u'; break;}return len;}
public boolean sameProperties(FontRecord other) {returnfield_1_font_height         == other.field_1_font_height &&field_2_attributes          == other.field_2_attributes &&field_3_color_palette_index == other.field_3_color_palette_index &&field_4_bold_weight         == other.field_4_bold_weight &&field_5_super_sub_script    == other.field_5_super_sub_script &&field_6_underline           == other.field_6_underline &&field_7_family              == other.field_7_family &&field_8_charset             == other.field_8_charset &&field_9_zero                == other.field_9_zero &&Objects.equals(this.field_11_font_name, other.field_11_font_name);}
public String toFormulaString() {return FormulaError.REF.getString();}
public StartTextDetectionResult startTextDetection(StartTextDetectionRequest request) {request = beforeClientExecution(request);return executeStartTextDetection(request);}
public DeleteMessageBatchRequestEntry(String id, String receiptHandle) {setId(id);setReceiptHandle(receiptHandle);}
public PatternCaptureGroupTokenFilter create(TokenStream input) {return new PatternCaptureGroupTokenFilter(input, preserveOriginal, pattern);}
public SigningCertificate(String userName, String certificateId, String certificateBody, StatusType status) {setUserName(userName);setCertificateId(certificateId);setCertificateBody(certificateBody);setStatus(status.toString());}
public DistributionConfig(String callerReference, Boolean enabled) {setCallerReference(callerReference);setEnabled(enabled);}
public FastCharStream(Reader r) {input = r;}
public int end(int group) {ensureMatch();return matchOffsets[(group * 2) + 1];}
public final Map.Entry<K, V> next() { return nextEntry(); }
public BlameCommand setTextComparator(RawTextComparator textComparator) {this.textComparator = textComparator;return this;}
public final T pop() {if (size > 0) {T result = heap[1];       heap[1] = heap[size];     heap[size] = null;        size--;downHeap(1);              return result;} else {return null;}}
public String toString() {return getClass().getSimpleName() + "(fields=" + fields.size() + ",delegate=" + postingsReader + ")";}
public static String shortenRefName(String noteRefName) {if (noteRefName.startsWith(Constants.R_NOTES))return noteRefName.substring(Constants.R_NOTES.length());return noteRefName;}
public DescribeDomainsResult describeDomains() {return describeDomains(new DescribeDomainsRequest());}
public int available() {return ccis.available();}
public GetContentModerationResult getContentModeration(GetContentModerationRequest request) {request = beforeClientExecution(request);return executeGetContentModeration(request);}
public PrintStream(OutputStream out) {super(out);if (out == null) {throw new NullPointerException();}}
public long ramBytesUsed() {long ramBytesUsed = postingsReader.ramBytesUsed();for (TermsReader r : fields.values()) {ramBytesUsed += r.ramBytesUsed();}return ramBytesUsed;}
public GetIntegrationResult getIntegration(GetIntegrationRequest request) {request = beforeClientExecution(request);return executeGetIntegration(request);}
public void setVisibility(int v) {if (getVisibility() != v) {super.setVisibility(v);if (mIndeterminate) {if (v == GONE || v == INVISIBLE) {stopAnimation();} else {startAnimation();}}}}
public boolean matches(char s[], int len) {if (!super.matches(s, len))return false;for (int i = 0; i < exceptions.length; i++)if (endsWith(s, len, exceptions[i]))return false;return true;}
public DescribeFleetCapacityResult describeFleetCapacity(DescribeFleetCapacityRequest request) {request = beforeClientExecution(request);return executeDescribeFleetCapacity(request);}
public UploadPackInternalServerErrorException(Throwable why) {initCause(why);}
public GetNetworkResult getNetwork(GetNetworkRequest request) {request = beforeClientExecution(request);return executeGetNetwork(request);}
public AllocatePrivateVirtualInterfaceResult allocatePrivateVirtualInterface(AllocatePrivateVirtualInterfaceRequest request) {request = beforeClientExecution(request);return executeAllocatePrivateVirtualInterface(request);}
public GetDeploymentResult getDeployment(GetDeploymentRequest request) {request = beforeClientExecution(request);return executeGetDeployment(request);}
public UpdateRepoAuthorizationRequest() {super("cr", "2016-06-07", "UpdateRepoAuthorization", "cr");setUriPattern("/repos/[RepoNamespace]/[RepoName]/authorizations/[AuthorizeId]");setMethod(MethodType.POST);}
public void foldToASCII(char[] input, int length){final int maxSizeNeeded = 4 * length;if (output.length < maxSizeNeeded) {output = new char[ArrayUtil.oversize(maxSizeNeeded, Character.BYTES)];}outputPos = foldToASCII(input, 0, output, 0, length);if (preserveOriginal && needToPreserve(input, length)) {state = captureState();}}
public boolean hasEntry(String name) {if (excludes.contains(name)) {return false;}return directory.hasEntry(name);}
public void setLockMessage(String msg) {lockMessage = msg;}
public ReflogCommand reflog() {return new ReflogCommand(repo);}
public void serialize(LittleEndianOutput out) {out.writeShort(getFirstRow());out.writeShort(getLastRow());out.writeShort(getFirstColumn());out.writeShort(getLastColumn());}
public static int response(java.net.HttpURLConnection c)throws IOException {try {return c.getResponseCode();} catch (ConnectException ce) {final URL url = c.getURL();final String host = (url == null) ? "<null>" : url.getHost(); if ("Connection timed out: connect".equals(ce.getMessage())) throw new ConnectException(MessageFormat.format(JGitText.get().connectionTimeOut, host));throw new ConnectException(ce.getMessage() + " " + host); }}
public static void fill(long[] array, long value) {for (int i = 0; i < array.length; i++) {array[i] = value;}}
public void serialize(LittleEndianOutput out) {out.writeInt(getPositionOfBof());out.writeShort(field_2_option_flags);String name = field_5_sheetname;out.writeByte(name.length());out.writeByte(field_4_isMultibyteUnicode);if (isMultibyte()) {StringUtil.putUnicodeLE(name, out);} else {StringUtil.putCompressedUnicode(name, out);}}
public static String getNonBlankTextOrFail(Element e) throws ParserException {String v = getText(e);if (null != v)v = v.trim();if (null == v || 0 == v.length()) {throw new ParserException(e.getTagName() + " has no text");}return v;}
public void buildFieldConfig(FieldConfig fieldConfig) {Map<String, Float> fieldBoostMap = this.config.get(ConfigurationKeys.FIELD_BOOST_MAP);if (fieldBoostMap != null) {Float boost = fieldBoostMap.get(fieldConfig.getField());if (boost != null) {fieldConfig.set(ConfigurationKeys.BOOST, boost);}}}
public PutLifecyclePolicyResult putLifecyclePolicy(PutLifecyclePolicyRequest request) {request = beforeClientExecution(request);return executePutLifecyclePolicy(request);}
public SortedSet<E> subSet(E start, E end) {return subSet(start, true, end, false);}
public void setParams(String params) {super.setParams(params);if (params != null) {int multiplier;if (params.endsWith("s")) {multiplier = 1;params = params.substring(0, params.length()-1);} else if (params.endsWith("m")) {multiplier = 60;params = params.substring(0, params.length()-1);} else if (params.endsWith("h")) {multiplier = 3600;params = params.substring(0, params.length()-1);} else {multiplier = 1;}waitTimeSec = Double.parseDouble(params) * multiplier;} else {throw new IllegalArgumentException("you must specify the wait time, eg: 10.0s, 4.5m, 2h");}}
public PutAttributesRequest(String domainName, String itemName, java.util.List<ReplaceableAttribute> attributes, UpdateCondition expected) {setDomainName(domainName);setItemName(itemName);setAttributes(attributes);setExpected(expected);}
public DescribeStreamConsumerResult describeStreamConsumer(DescribeStreamConsumerRequest request) {request = beforeClientExecution(request);return executeDescribeStreamConsumer(request);}
public void freeze() {this.frozen = true;}
public FuzzyLikeThisQueryBuilder(Analyzer analyzer) {this.analyzer = analyzer;}
public DBClusterSnapshot copyDBClusterSnapshot(CopyDBClusterSnapshotRequest request) {request = beforeClientExecution(request);return executeCopyDBClusterSnapshot(request);}
public OutputStreamDataOutput(OutputStream os) {this.os = os;}
public String findPattern(String pat) {int k = super.find(pat);if (k >= 0) {return unpackValues(k);}return "";}
public static int murmurhash3_x86_32(BytesRef bytes, int seed) {return murmurhash3_x86_32(bytes.bytes, bytes.offset, bytes.length, seed);}
public boolean isOverridable() {return overridable;}
public UpdateMemberResult updateMember(UpdateMemberRequest request) {request = beforeClientExecution(request);return executeUpdateMember(request);}
public CopyFpgaImageResult copyFpgaImage(CopyFpgaImageRequest request) {request = beforeClientExecution(request);return executeCopyFpgaImage(request);}
public void inform(ResourceLoader loader) {try { OpenNLPOpsFactory.getPOSTaggerModel(posTaggerModelFile, loader);} catch (IOException e) {throw new IllegalArgumentException(e);}}
public CellRangeAddress(int firstRow, int lastRow, int firstCol, int lastCol) {super(firstRow, lastRow, firstCol, lastCol);if (lastRow < firstRow || lastCol < firstCol) {throw new IllegalArgumentException("Invalid cell range, having lastRow < firstRow || lastCol < firstCol, " +"had rows " + lastRow + " >= " + firstRow + " or cells " + lastCol + " >= " + firstCol);}}
public boolean equals(ATNConfig a, ATNConfig b) {if ( a==b ) return true;if ( a==null || b==null ) return false;return a.state.stateNumber==b.state.stateNumber&& a.context.equals(b.context);}
public PushCommand setPushTags() {refSpecs.add(Transport.REFSPEC_TAGS);return this;}
public CreateEvaluationResult createEvaluation(CreateEvaluationRequest request) {request = beforeClientExecution(request);return executeCreateEvaluation(request);}
public DescribeOrderableDBInstanceOptionsResult describeOrderableDBInstanceOptions(DescribeOrderableDBInstanceOptionsRequest request) {request = beforeClientExecution(request);return executeDescribeOrderableDBInstanceOptions(request);}
public long getPosition() {return (long) currentBlockIndex * blockSize + currentBlockUpto;}
public TokenStream create(TokenStream input) {return new FrenchLightStemFilter(input);}
public AssignPrivateIpAddressesResult assignPrivateIpAddresses(AssignPrivateIpAddressesRequest request) {request = beforeClientExecution(request);return executeAssignPrivateIpAddresses(request);}
public boolean setExecute(File f, boolean canExec) {return false;}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval lookup_value, ValueEval table_array,ValueEval col_index, ValueEval range_lookup) {try {ValueEval lookupValue = OperandResolver.getSingleValue(lookup_value, srcRowIndex, srcColumnIndex);TwoDEval tableArray = LookupUtils.resolveTableArrayArg(table_array);boolean isRangeLookup;try {isRangeLookup = LookupUtils.resolveRangeLookupArg(range_lookup, srcRowIndex, srcColumnIndex);} catch(RuntimeException e) {isRangeLookup = true;}int rowIndex = LookupUtils.lookupIndexOfValue(lookupValue, LookupUtils.createColumnVector(tableArray, 0), isRangeLookup);int colIndex = LookupUtils.resolveRowOrColIndexArg(col_index, srcRowIndex, srcColumnIndex);ValueVector resultCol = createResultColumnVector(tableArray, colIndex);return resultCol.getItem(rowIndex);} catch (EvaluationException e) {return e.getErrorEval();}}
public CreateGameSessionResult createGameSession(CreateGameSessionRequest request) {request = beforeClientExecution(request);return executeCreateGameSession(request);}
public RowRecord getRow(int rowIndex) {int maxrow = SpreadsheetVersion.EXCEL97.getLastRowIndex();if (rowIndex < 0 || rowIndex > maxrow) {throw new IllegalArgumentException("The row number must be between 0 and " + maxrow + ", but had: " + rowIndex);}return _rowRecords.get(Integer.valueOf(rowIndex));}
public DescribeClientPropertiesResult describeClientProperties(DescribeClientPropertiesRequest request) {request = beforeClientExecution(request);return executeDescribeClientProperties(request);}
public Builder(CompositeReader reader) {this.reader = reader;}
public synchronized void mark(int readlimit) {in.mark(readlimit);}
public void print(int inum) {print(String.valueOf(inum));}
public static final ObjectId fromRaw(int[] is) {return fromRaw(is, 0);}
public String toString() {return slice.toString()+":"+ postingsEnum;}
public void serialize(LittleEndianOutput out) {out.writeShort(getMode());}
@Override public int size() {return Impl.this.size();}
public static int hashCode(Object... objects) {return Arrays.hashCode(objects);}
public ByteBuffer putFloat(int index, float value) {throw new ReadOnlyBufferException();}
public ListJournalS3ExportsForLedgerResult listJournalS3ExportsForLedger(ListJournalS3ExportsForLedgerRequest request) {request = beforeClientExecution(request);return executeListJournalS3ExportsForLedger(request);}
public DeleteMessageBatchResult deleteMessageBatch(DeleteMessageBatchRequest request) {request = beforeClientExecution(request);return executeDeleteMessageBatch(request);}
public final void write(LittleEndianOutput out) {out.writeByte(getSid() + getPtgClass());writeCoordinates(out);}
public FSTCompletionBuilder(int buckets, BytesRefSorter sorter, int shareMaxTailLength) {if (buckets < 1 || buckets > 255) {throw new IllegalArgumentException("Buckets must be >= 1 and <= 255: "+ buckets);}if (sorter == null) throw new IllegalArgumentException("BytesRefSorter must not be null.");this.sorter = sorter;this.buckets = buckets;this.shareMaxTailLength = shareMaxTailLength;}
public void incRef() {refCount.incrementAndGet();}
public boolean supports(CredentialItem... items) {for (CredentialItem i : items) {if (i instanceof CredentialItem.Username)continue;else if (i instanceof CredentialItem.Password)continue;elsereturn false;}return true;}
public DeleteVpnConnectionRequest(String vpnConnectionId) {setVpnConnectionId(vpnConnectionId);}
public final ValueEval evaluate(ValueEval[] args, int srcRowIndex, int srcColumnIndex) {if (args.length != 4) {return ErrorEval.VALUE_INVALID;}return evaluate(srcRowIndex, srcColumnIndex, args[0], args[1], args[2], args[3]);}
public void print(double d) {print(String.valueOf(d));}
public UpdateUserProfileResult updateUserProfile(UpdateUserProfileRequest request) {request = beforeClientExecution(request);return executeUpdateUserProfile(request);}
public RevFilter clone() {final RevFilter[] s = new RevFilter[subfilters.length];for (int i = 0; i < s.length; i++)s[i] = subfilters[i].clone();return new List(s);}
public GetFederationTokenRequest(String name) {setName(name);}
public static Cell translateUnicodeValues(Cell cell) {String s = cell.getRichStringCellValue().getString();boolean foundUnicode = false;String lowerCaseStr = s.toLowerCase(Locale.ROOT);for (UnicodeMapping entry : unicodeMappings) {String key = entry.entityName;if (lowerCaseStr.contains(key)) {s = s.replaceAll(key, entry.resolvedValue);foundUnicode = true;}}if (foundUnicode) {cell.setCellValue(cell.getRow().getSheet().getWorkbook().getCreationHelper().createRichTextString(s));}return cell;}
public CreateChangeSetResult createChangeSet(CreateChangeSetRequest request) {request = beforeClientExecution(request);return executeCreateChangeSet(request);}
public SubmoduleStatusCommand(Repository repo) {super(repo);paths = new ArrayList<>();}
public int getResultStart() {return outRegion.resultStart;}
public static BigInteger round(BigInteger bi, int nBits) {if (nBits < 1) {return bi;}return bi.add(HALF_BITS[nBits]);}
public static Date round(Date date, Resolution resolution) {return new Date(round(date.getTime(), resolution));}
public static int compareArrayByPrefix(char[] shortArray, int shortIndex,char[] longArray, int longIndex) {if (shortArray == null)return 0;else if (longArray == null)return (shortIndex < shortArray.length) ? 1 : 0;int si = shortIndex, li = longIndex;while (si < shortArray.length && li < longArray.length&& shortArray[si] == longArray[li]) {si++;li++;}if (si == shortArray.length) {return 0;} else {if (li == longArray.length)return 1;elsereturn (shortArray[si] > longArray[li]) ? 1 : -1;}}
public AttachInternetGatewayResult attachInternetGateway(AttachInternetGatewayRequest request) {request = beforeClientExecution(request);return executeAttachInternetGateway(request);}
public synchronized boolean containsValue(Object value) {if (value == null) {throw new NullPointerException();}HashtableEntry[] tab = table;int len = tab.length;for (int i = 0; i < len; i++) {for (HashtableEntry e = tab[i]; e != null; e = e.next) {if (value.equals(e.value)) {return true;}}}return false;}
public String toFormulaString(String[] operands) {StringBuilder buffer = new StringBuilder();buffer.append( operands[0] );buffer.append("<=");buffer.append( operands[1] );return buffer.toString();}
public void write(String str) {write(str.toCharArray());}
public Sort(SortField... fields) {setSort(fields);}
public DescribeEventCategoriesResult describeEventCategories(DescribeEventCategoriesRequest request) {request = beforeClientExecution(request);return executeDescribeEventCategories(request);}
public UpdateDeviceResult updateDevice(UpdateDeviceRequest request) {request = beforeClientExecution(request);return executeUpdateDevice(request);}
public CreateWorkerBlockResult createWorkerBlock(CreateWorkerBlockRequest request) {request = beforeClientExecution(request);return executeCreateWorkerBlock(request);}
public synchronized void reset() throws IOException {throw new IOException();}
public final void setReader(Reader input) {if (input == null) {throw new NullPointerException("input must not be null");} else if (this.input != ILLEGAL_STATE_READER) {throw new IllegalStateException("TokenStream contract violation: close() call missing");}this.inputPending = input;setReaderTestPoint();}
public GetUsagePlanKeysResult getUsagePlanKeys(GetUsagePlanKeysRequest request) {request = beforeClientExecution(request);return executeGetUsagePlanKeys(request);}
public String toString(){StringBuilder sb = new StringBuilder();sb.append( "subInfos=(" );for( SubInfo si : subInfos )sb.append( si.toString() );sb.append( ")/" ).append( totalBoost ).append( '(' ).append( startOffset ).append( ',' ).append( endOffset ).append( ')' );return sb.toString();}
public TokenStream create(TokenStream input) {return new LimitTokenPositionFilter(input, maxTokenPosition, consumeAllTokens);}
public DescribeFleetUtilizationResult describeFleetUtilization(DescribeFleetUtilizationRequest request) {request = beforeClientExecution(request);return executeDescribeFleetUtilization(request);}
public void inform(ResourceLoader loader) throws IOException {InputStream stream = null;try {if (dictFile != null) dictionary = getWordSet(loader, dictFile, false);stream = loader.openResource(hypFile);final InputSource is = new InputSource(stream);is.setEncoding(encoding); is.setSystemId(hypFile);hyphenator = HyphenationCompoundWordTokenFilter.getHyphenationTree(is);} finally {IOUtils.closeWhileHandlingException(stream);}}
public DeclineInvitationsResult declineInvitations(DeclineInvitationsRequest request) {request = beforeClientExecution(request);return executeDeclineInvitations(request);}
public DescribeAutoScalingGroupsResult describeAutoScalingGroups() {return describeAutoScalingGroups(new DescribeAutoScalingGroupsRequest());}
public String toString() {return String.format("pushMode(%d)", mode);}
public CreateBranchCommand setStartPoint(String startPoint) {checkCallable();this.startPoint = startPoint;this.startCommit = null;return this;}
public DBInstance stopDBInstance(StopDBInstanceRequest request) {request = beforeClientExecution(request);return executeStopDBInstance(request);}
public SuggestWordQueue(int size, Comparator<SuggestWord> comparator){super(size);this.comparator = comparator;}
public LBCookieStickinessPolicy(String policyName, Long cookieExpirationPeriod) {setPolicyName(policyName);setCookieExpirationPeriod(cookieExpirationPeriod);}
public SheetRangeEvaluator(int firstSheetIndex, int lastSheetIndex, SheetRefEvaluator[] sheetEvaluators) {if (firstSheetIndex < 0) {throw new IllegalArgumentException("Invalid firstSheetIndex: " + firstSheetIndex + ".");}if (lastSheetIndex < firstSheetIndex) {throw new IllegalArgumentException("Invalid lastSheetIndex: " + lastSheetIndex + " for firstSheetIndex: " + firstSheetIndex + ".");}_firstSheetIndex = firstSheetIndex;_lastSheetIndex = lastSheetIndex;_sheetEvaluators = sheetEvaluators.clone();}
public RevokeTokenRequest() {super("OnsMqtt", "2019-12-11", "RevokeToken", "onsmqtt");setMethod(MethodType.POST);}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval arg0, ValueEval arg1) {try {ValueEval ve = OperandResolver.getSingleValue(arg0, srcRowIndex, srcColumnIndex);double result = OperandResolver.coerceValueToDouble(ve);if (Double.isNaN(result) || Double.isInfinite(result)) {throw new EvaluationException(ErrorEval.NUM_ERROR);}if (arg1 instanceof RefListEval) {return eval(result, ((RefListEval)arg1), true);}final AreaEval aeRange = convertRangeArg(arg1);return eval(result, aeRange, true);} catch (EvaluationException e) {return e.getErrorEval();}}
public String toFormulaString() {return "";}
public byte readByte() throws IOException {if (bufferPos == bufferSize) {refill();}assert bufferPos == buffer.position() : "bufferPos=" + bufferPos + " vs buffer.position()=" + buffer.position();bufferPos++;return buffer.get();}
public ListTargetsByRuleResult listTargetsByRule(ListTargetsByRuleRequest request) {request = beforeClientExecution(request);return executeListTargetsByRule(request);}
public DisassociateQualificationFromWorkerResult disassociateQualificationFromWorker(DisassociateQualificationFromWorkerRequest request) {request = beforeClientExecution(request);return executeDisassociateQualificationFromWorker(request);}
public boolean equals(Object obj) {if (this == obj) return true;if (obj == null) return false;if (getClass() != obj.getClass()) return false;CompiledAutomaton other = (CompiledAutomaton) obj;if (type != other.type) return false;if (type == AUTOMATON_TYPE.SINGLE) {if (!term.equals(other.term)) return false;} else if (type == AUTOMATON_TYPE.NORMAL) {if (!runAutomaton.equals(other.runAutomaton)) return false;}return true;}
public static CharFilterFactory forName(String name, Map<String,String> args) {return loader.newInstance(name, args);}
public String toString() {String[] units = { "bytes", "KiB", "MiB", "GiB" };long sz = getIndexSize();int u = 0;while (1024 <= sz && u < units.length - 1) {int rem = (int) (sz % 1024);sz /= 1024;if (rem != 0)sz++;u++;}return "DeltaIndex[" + sz + " " + units[u] + "]";}
public SimilarityConfig build() {return new SimilarityConfig(this);}
public void mark(int readLimit) throws IOException {throw new IOException();}
public void collect(int doc) throws IOException {final long time = clock.get();if (time - timeout > 0L) {if (greedy) {in.collect(doc);}throw new TimeExceededException( timeout-t0, time-t0, docBase + doc );}in.collect(doc);}
public LocalFile(File directory, int inCoreLimit) {super(inCoreLimit);this.directory = directory;}
@Override public E remove(int index) {Object[] a = array;int s = size;if (index >= s) {throwIndexOutOfBoundsException(index, s);}@SuppressWarnings("unchecked") E result = (E) a[index];System.arraycopy(a, index + 1, a, index, --s - index);a[s] = null;  size = s;modCount++;return result;}
public RequestUploadCredentialsResult requestUploadCredentials(RequestUploadCredentialsRequest request) {request = beforeClientExecution(request);return executeRequestUploadCredentials(request);}
public void copyTo(OutputStream out) throws MissingObjectException,IOException {if (isLarge()) {try (ObjectStream in = openStream()) {final long sz = in.getSize();byte[] tmp = new byte[8192];long copied = 0;while (copied < sz) {int n = in.read(tmp);if (n < 0)throw new EOFException();out.write(tmp, 0, n);copied += n;}if (0 <= in.read())throw new EOFException();}} else {out.write(getCachedBytes());}}
@Override public V remove(Object key) {if (key == null) {return removeNullKey();}int hash = secondaryHash(key.hashCode());HashMapEntry<K, V>[] tab = table;int index = hash & (tab.length - 1);for (HashMapEntry<K, V> e = tab[index], prev = null;e != null; prev = e, e = e.next) {if (e.hash == hash && key.equals(e.key)) {if (prev == null) {tab[index] = e.next;} else {prev.next = e.next;}modCount++;size--;postRemove(e);return e.value;}}return null;}
public RevFilter negate() {return a;}
public DescribeVpcsResult describeVpcs(DescribeVpcsRequest request) {request = beforeClientExecution(request);return executeDescribeVpcs(request);}
public UpdateGameSessionQueueResult updateGameSessionQueue(UpdateGameSessionQueueRequest request) {request = beforeClientExecution(request);return executeUpdateGameSessionQueue(request);}
public String getTitle() {return title;}
public final void setNewHeads(List<Head> newHeads) {if (this.newHeads != null)throw new IllegalStateException(JGitText.get().propertyIsAlreadyNonNull);this.newHeads = newHeads;}
public ObjectId getExpectedOldObjectId() {return expectedOldObjectId;}
public GetRecordsResult getRecords(GetRecordsRequest request) {request = beforeClientExecution(request);return executeGetRecords(request);}
public Deleted3DPxg(int externalWorkbookNumber, String sheetName) {this.externalWorkbookNumber = externalWorkbookNumber;this.sheetName = sheetName;}
public void execute(Lexer lexer) {lexer.skip();}
public DescribeScheduledInstancesResult describeScheduledInstances(DescribeScheduledInstancesRequest request) {request = beforeClientExecution(request);return executeDescribeScheduledInstances(request);}
public MultiFields(Fields[] subs, ReaderSlice[] subSlices) {this.subs = subs;this.subSlices = subSlices;}
public int peekNextSid() {if(!hasNext()) {return -1;}return _list.get(_nextIndex).getSid();}
public ConfigureAgentResult configureAgent(ConfigureAgentRequest request) {request = beforeClientExecution(request);return executeConfigureAgent(request);}
public GetStreamingDistributionResult getStreamingDistribution(GetStreamingDistributionRequest request) {request = beforeClientExecution(request);return executeGetStreamingDistribution(request);}
public ListTrialComponentsResult listTrialComponents(ListTrialComponentsRequest request) {request = beforeClientExecution(request);return executeListTrialComponents(request);}
public ByteBuffer putShort(int index, short value) {throw new ReadOnlyBufferException();}
public int compareNormalised(NormalisedDecimal other) {int cmp = _relativeDecimalExponent - other._relativeDecimalExponent;if (cmp != 0) {return cmp;}if (_wholePart > other._wholePart) {return 1;}if (_wholePart < other._wholePart) {return -1;}return _fractionalPart - other._fractionalPart;}
public TokenStream create(TokenStream input) {return new JapaneseKatakanaStemFilter(input, minimumLength);}
public EnableAvailabilityZonesForLoadBalancerResult enableAvailabilityZonesForLoadBalancer(EnableAvailabilityZonesForLoadBalancerRequest request) {request = beforeClientExecution(request);return executeEnableAvailabilityZonesForLoadBalancer(request);}
public UpdateEnvironmentResult updateEnvironment(UpdateEnvironmentRequest request) {request = beforeClientExecution(request);return executeUpdateEnvironment(request);}
public ListTagsForDomainResult listTagsForDomain(ListTagsForDomainRequest request) {request = beforeClientExecution(request);return executeListTagsForDomain(request);}
public static double log(double base, double x) {return Math.log(x) / Math.log(base);}
public final void writeBoolean(boolean val) throws IOException {out.write(val ? 1 : 0);written++;}
public boolean equals(Object other) {if (!(other instanceof ByteBuffer)) {return false;}ByteBuffer otherBuffer = (ByteBuffer) other;if (remaining() != otherBuffer.remaining()) {return false;}int myPosition = position;int otherPosition = otherBuffer.position;boolean equalSoFar = true;while (equalSoFar && (myPosition < limit)) {equalSoFar = get(myPosition++) == otherBuffer.get(otherPosition++);}return equalSoFar;}
public DescribeVirtualGatewaysResult describeVirtualGateways() {return describeVirtualGateways(new DescribeVirtualGatewaysRequest());}
public FieldConfig getFieldConfig(String fieldName) {FieldConfig fieldConfig = new FieldConfig(StringUtils.toString(fieldName));for (FieldConfigListener listener : this.listeners) {listener.buildFieldConfig(fieldConfig);}return fieldConfig;}
public void setProperty(Row row, int column) {Cell cell = CellUtil.getCell(row, column);CellUtil.setCellStyleProperty(cell, _propertyName, _propertyValue);}
public RebootInstancesResult rebootInstances(RebootInstancesRequest request) {request = beforeClientExecution(request);return executeRebootInstances(request);}
public Predicate(int ruleIndex, int predIndex, boolean isCtxDependent) {this.ruleIndex = ruleIndex;this.predIndex = predIndex;this.isCtxDependent = isCtxDependent;}
public void fillPolygon(int[] xPoints, int[] yPoints,int nPoints){int right  = findBiggest(xPoints);int bottom = findBiggest(yPoints);int left   = findSmallest(xPoints);int top    = findSmallest(yPoints);HSSFPolygon shape = escherGroup.createPolygon(new HSSFChildAnchor(left,top,right,bottom) );shape.setPolygonDrawArea(right - left, bottom - top);shape.setPoints(addToAll(xPoints, -left), addToAll(yPoints, -top));shape.setLineStyleColor(foreground.getRed(), foreground.getGreen(), foreground.getBlue());shape.setFillColor(foreground.getRed(), foreground.getGreen(), foreground.getBlue());}
public ListEventsRequest() {super("Status", "2020-01-17", "ListEvents", "StatusAPI");setMethod(MethodType.POST);}
public ListIAMPolicyAssignmentsResult listIAMPolicyAssignments(ListIAMPolicyAssignmentsRequest request) {request = beforeClientExecution(request);return executeListIAMPolicyAssignments(request);}
public CountingOutputStream(OutputStream out) {this.out = out;}
public void seekExact(BytesRef target, TermState otherState) {if (!target.equals(term)) {state.copyFrom(otherState);term = BytesRef.deepCopyOf(target);seekPending = true;}}
public void seek(long pos) throws IOException {if (pos != getFilePointer()) {final long alignedPos = pos & ALIGN_NOT_MASK;filePos = alignedPos-bufferSize;final int delta = (int) (pos - alignedPos);if (delta != 0) {refill();buffer.position(delta);bufferPos = delta;} else {bufferPos = bufferSize;}}}
public void clear() {removeAllElements();}
public QueryCustomerByPhoneRequest() {super("xspace", "2017-07-20", "QueryCustomerByPhone");setUriPattern("/customerbyphone");setMethod(MethodType.POST);}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval arg0) {return this.evaluate(srcRowIndex, srcColumnIndex, arg0, null);}
public ListDashboardVersionsResult listDashboardVersions(ListDashboardVersionsRequest request) {request = beforeClientExecution(request);return executeListDashboardVersions(request);}
public IntBuffer put(int c) {if (position == limit) {throw new BufferOverflowException();}backingArray[offset + position++] = c;return this;}
public DeleteHostedZoneResult deleteHostedZone(DeleteHostedZoneRequest request) {request = beforeClientExecution(request);return executeDeleteHostedZone(request);}
public CreateReceiptRuleResult createReceiptRule(CreateReceiptRuleRequest request) {request = beforeClientExecution(request);return executeCreateReceiptRule(request);}
public Result rename() throws IOException {try {result = doRename();return result;} catch (IOException err) {result = Result.IO_FAILURE;throw err;}}
public DescribeDBInstancesResult describeDBInstances() {return describeDBInstances(new DescribeDBInstancesRequest());}
public String toString() {if (label != null) {return label + ":" + tag;}return tag;}
public CharSequence toQueryString(EscapeQuerySyntax escaper) {return "[DELETEDCHILD]";}
public CreateAccountResult createAccount(CreateAccountRequest request) {request = beforeClientExecution(request);return executeCreateAccount(request);}
public Map.Entry<K,V> next() {HashEntry<K,V> e = super.nextEntry();return new WriteThroughEntry(e.key, e.value);}
public BaseRef(RefEval re) {_refEval = re;_areaEval = null;_firstRowIndex = re.getRow();_firstColumnIndex = re.getColumn();_height = 1;_width = 1;}
public void decode(long[] blocks, int blocksOffset, long[] values, int valuesOffset, int iterations) {for (int i = 0; i < iterations; ++i) {final long block = blocks[blocksOffset++];for (int shift = 62; shift >= 0; shift -= 2) {values[valuesOffset++] = (block >>> shift) & 3;}}}
public void unrollRecursionContexts(ParserRuleContext _parentctx) {_precedenceStack.pop();_ctx.stop = _input.LT(-1);ParserRuleContext retctx = _ctx; if ( _parseListeners != null ) {while ( _ctx != _parentctx ) {triggerExitRuleEvent();_ctx = (ParserRuleContext)_ctx.parent;}}else {_ctx = _parentctx;}retctx.parent = _parentctx;if (_buildParseTrees && _parentctx != null) {_parentctx.addChild(retctx);}}
public CancelBundleTaskRequest(String bundleId) {setBundleId(bundleId);}
public void add(CharsRef input, CharsRef output, boolean includeOrig) {add(input, countWords(input), output, countWords(output), includeOrig);}
public SetIdentityDkimEnabledResult setIdentityDkimEnabled(SetIdentityDkimEnabledRequest request) {request = beforeClientExecution(request);return executeSetIdentityDkimEnabled(request);}
public GetResolverEndpointResult getResolverEndpoint(GetResolverEndpointRequest request) {request = beforeClientExecution(request);return executeGetResolverEndpoint(request);}
public void setText(String value) {string = value;start = offset = 0;end = value.length();}
public String toString() {return toString(0);}
public void adjustIndex(int offset) {_firstSheetIndex += offset;_lastSheetIndex += offset;}
public GalicianStemFilterFactory(Map<String,String> args) {super(args);if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public ListRepositoryAssociationsResult listRepositoryAssociations(ListRepositoryAssociationsRequest request) {request = beforeClientExecution(request);return executeListRepositoryAssociations(request);}
public void setParams(String params) {super.setParams(params);maxNumSegments = (int)Double.parseDouble(params);}
public char getChar() {return (char) getShort();}
public void next(int delta) {if (delta == 1) {prevPtr = currPtr;currPtr = nextPtr;if (!eof())parseEntry();return;}final int end = raw.length;int ptr = nextPtr;while (--delta > 0 && ptr != end) {prevPtr = ptr;while (raw[ptr] != 0)ptr++;ptr += OBJECT_ID_LENGTH + 1;}if (delta != 0)throw new ArrayIndexOutOfBoundsException(delta);currPtr = ptr;if (!eof())parseEntry();}
public Type getType() {return type;}
public CharBuffer duplicate() {return copy(this, mark);}
public NGramFilterFactory(Map<String, String> args) {super(args);minGramSize = requireInt(args, "minGramSize");maxGramSize = requireInt(args, "maxGramSize");preserveOriginal = getBoolean(args, "preserveOriginal", NGramTokenFilter.DEFAULT_PRESERVE_ORIGINAL);if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public AddRoleToDBClusterResult addRoleToDBCluster(AddRoleToDBClusterRequest request) {request = beforeClientExecution(request);return executeAddRoleToDBCluster(request);}
public BlameGenerator setTextComparator(RawTextComparator comparator) {textComparator = comparator;return this;}
public  PatternCaptureGroupFilterFactory(Map<String,String> args) {super(args);pattern = getPattern(args, "pattern");preserveOriginal = args.containsKey("preserve_original") ? Boolean.parseBoolean(args.get("preserve_original")) : true;}
public CreateObjectResult createObject(CreateObjectRequest request) {request = beforeClientExecution(request);return executeCreateObject(request);}
@Override public String getActions() { return null; }
public void onChanged() {if (mAdapter != null) {post(new Runnable() {@Override
public CreateResourceGroupResult createResourceGroup(CreateResourceGroupRequest request) {request = beforeClientExecution(request);return executeCreateResourceGroup(request);}
public static RevFilter has(RevFlag a) {final RevFlagSet s = new RevFlagSet();s.add(a);return new HasAll(s);}
@Override public int size() {return totalSize;}
public void write(LittleEndianOutput out) {out.writeByte(sid + getPtgClass());out.writeShort(field_1_index_extern_sheet);out.writeInt(unused1);}
public String toString() {return this.getClass().getSimpleName() + "@" + directory + " lockFactory=" + lockFactory;}
public final ValueEval evaluate(ValueEval[] args, int srcRowIndex, int srcColumnIndex) {switch (args.length) {case 3:return evaluate(srcRowIndex, srcColumnIndex, args[0], args[1], args[2]);case 4:return evaluate(srcRowIndex, srcColumnIndex, args[0], args[1], args[2], args[3]);}return ErrorEval.VALUE_INVALID;}
public CancelDataRepositoryTaskResult cancelDataRepositoryTask(CancelDataRepositoryTaskRequest request) {request = beforeClientExecution(request);return executeCancelDataRepositoryTask(request);}
public DateFormatTokenizer(String format) {this.format = format;}
public static int getBiasedExponent(long rawBits) {return Math.toIntExact((rawBits & EXPONENT_MASK) >> EXPONENT_SHIFT);}
public String toString() {return "IB " + distribution.toString() + "-" + lambda.toString()+ normalization.toString();}
public String getName() {return name;}
public boolean inContext(String context) {return false;}
public String toString() {String desc;File directory = getDirectory();if (directory != null)desc = directory.getPath();elsedesc = getClass().getSimpleName() + "-" + System.identityHashCode(this);return "Repository[" + desc + "]"; }
public int serialize(int offset, byte [] data) {LittleEndian.putInt(data, offset+0, field_13_border_styles1);LittleEndian.putInt(data, offset+4, field_14_border_styles2);return 8;}
public void decode(byte[] blocks, int blocksOffset, long[] values, int valuesOffset, int iterations) {for (int j = 0; j < iterations; ++j) {final byte block = blocks[blocksOffset++];values[valuesOffset++] = (block >>> 7) & 1;values[valuesOffset++] = (block >>> 6) & 1;values[valuesOffset++] = (block >>> 5) & 1;values[valuesOffset++] = (block >>> 4) & 1;values[valuesOffset++] = (block >>> 3) & 1;values[valuesOffset++] = (block >>> 2) & 1;values[valuesOffset++] = (block >>> 1) & 1;values[valuesOffset++] = block & 1;}}
public PipedWriter(PipedReader destination) throws IOException {super(destination);connect(destination);}
public String dequote(byte[] in, int ip, int ie) {boolean inquote = false;final byte[] r = new byte[ie - ip];int rPtr = 0;while (ip < ie) {final byte b = in[ip++];switch (b) {case '\'':inquote = !inquote;continue;case '\\':if (inquote || ip == ie)r[rPtr++] = b; elser[rPtr++] = in[ip++];continue;default:r[rPtr++] = b;continue;}}return RawParseUtils.decode(UTF_8, r, 0, rPtr);}
public Status getStatus() {return myStatus;}
public DeltaRecord(RecordInputStream in) {field_1_max_change = in.readDouble();}
public void serialize(LittleEndianOutput out) {out.writeShort(getCount());}
public ListPartsRequest(String vaultName, String uploadId) {setVaultName(vaultName);setUploadId(uploadId);}
public void set(int index, long value) {final int o = index >>> 2;final int b = index & 3;final int shift = b << 4;blocks[o] = (blocks[o] & ~(65535L << shift)) | (value << shift);}
public void setRunInBackground(int deltaPri) {runInBackground = true;this.deltaPri = deltaPri;}
public TeeInputStream(InputStream src, OutputStream dst) {this.src = src;this.dst = dst;}
public void addChild(final Property property)throws IOException{String name = property.getName();if (_children_names.contains(name)){throw new IOException("Duplicate name \"" + name + "\"");}_children_names.add(name);_children.add(property);}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval arg0) {int result;if (arg0 instanceof TwoDEval) {result = ((TwoDEval) arg0).getWidth();} else if (arg0 instanceof RefEval) {result = 1;} else { return ErrorEval.VALUE_INVALID;}return new NumberEval(result);}
public ListModelsResult listModels(ListModelsRequest request) {request = beforeClientExecution(request);return executeListModels(request);}
public ExtensionQuery(QueryParser topLevelParser, String field, String rawQueryString) {this.field = field;this.rawQueryString = rawQueryString;this.topLevelParser = topLevelParser;}
public String toString() {return resourceDescription;}
public GetDeploymentInstanceResult getDeploymentInstance(GetDeploymentInstanceRequest request) {request = beforeClientExecution(request);return executeGetDeploymentInstance(request);}
public MappingCharFilterFactory(Map<String,String> args) {super(args);mapping = get(args, "mapping");if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public boolean promptPassphrase(String msg) {CredentialItem.StringType v = newPrompt(msg);if (provider.get(uri, v)) {passphrase = v.getValue();return true;}passphrase = null;return false;}
public DescribeReservedDBInstancesResult describeReservedDBInstances() {return describeReservedDBInstances(new DescribeReservedDBInstancesRequest());}
public UnsubscribeFromDatasetResult unsubscribeFromDataset(UnsubscribeFromDatasetRequest request) {request = beforeClientExecution(request);return executeUnsubscribeFromDataset(request);}
public int available() throws IOException {if (buf == null) {throw new IOException();}return buf.length - pos + in.available();}
@Override public V remove(Object key) {return isInBounds(key) ? TreeMap.this.remove(key) : null;}
public void insertSST() {LOG.log(DEBUG, "creating new SST via insertSST!");sst = new SSTRecord();records.add(records.size() - 1, createExtendedSST());records.add(records.size() - 2, sst);}
public AddApplicationCloudWatchLoggingOptionResult addApplicationCloudWatchLoggingOption(AddApplicationCloudWatchLoggingOptionRequest request) {request = beforeClientExecution(request);return executeAddApplicationCloudWatchLoggingOption(request);}
public ListCampaignsResult listCampaigns(ListCampaignsRequest request) {request = beforeClientExecution(request);return executeListCampaigns(request);}
public void execute(Lexer lexer) {lexer.more();}
public SetFaceCoverRequest() {super("CloudPhoto", "2017-07-11", "SetFaceCover", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public GetInstanceAccessResult getInstanceAccess(GetInstanceAccessRequest request) {request = beforeClientExecution(request);return executeGetInstanceAccess(request);}
public void clear() {value = null;}
public GetFederationTokenResult getFederationToken(GetFederationTokenRequest request) {request = beforeClientExecution(request);return executeGetFederationToken(request);}
public int first() {currentSentence = 0;text.setIndex(text.getBeginIndex());return current();}
public QueryPhraseMap getFieldTermMap( String fieldName, String term ){QueryPhraseMap rootMap = getRootMap( fieldName );return rootMap == null ? null : rootMap.subMap.get( term );}
@Override public boolean contains(Object object) {if (object instanceof Multiset.Entry) {Multiset.Entry<?> entry = (Multiset.Entry<?>) object;Object element = entry.getElement();int entryCount = entry.getCount();return entryCount > 0 && count(element) == entryCount;}return false;}
public DeleteLexiconResult deleteLexicon(DeleteLexiconRequest request) {request = beforeClientExecution(request);return executeDeleteLexicon(request);}
public DomainMetadataResult domainMetadata(DomainMetadataRequest request) {request = beforeClientExecution(request);return executeDomainMetadata(request);}
public RevFlag getReinterestingFlag() {return REINTERESTING;}
public static void advise(FileDescriptor fd, long offset, long len, int advise) throws IOException {final int code = posix_fadvise(fd, offset, len, advise);if (code != 0) {throw new RuntimeException("posix_fadvise failed code=" + code);}}
public DeleteSchemaResult deleteSchema(DeleteSchemaRequest request) {request = beforeClientExecution(request);return executeDeleteSchema(request);}
public CreateBatchInferenceJobResult createBatchInferenceJob(CreateBatchInferenceJobRequest request) {request = beforeClientExecution(request);return executeCreateBatchInferenceJob(request);}
public BitField(final int mask){_mask = mask;int count       = 0;int bit_pattern = mask;if (bit_pattern != 0){while ((bit_pattern & 1) == 0){count++;bit_pattern >>= 1;}}_shift_count = count;}
public boolean failed() {return !failingPaths.isEmpty();}
public String toString() {StringBuilder b = new StringBuilder();for(int i=0;i<len;i++) {if (i > 0) {b.append(' ');}b.append(Integer.toBinaryString(bytes[i].value));}return b.toString();}
public final void remove() {if (modCount != expectedModCount)throw new ConcurrentModificationException();if (lastReturned == null)throw new IllegalStateException();LinkedHashMap.this.remove(lastReturned.key);lastReturned = null;expectedModCount = modCount;}
public boolean shouldBeRecursive() {return path.shouldBeRecursive() || ANY_DIFF.shouldBeRecursive();}
public DeleteQueueRequest(String queueUrl) {setQueueUrl(queueUrl);}
public ExternalName getExternalName(int externSheetIndex, int externNameIndex) {String nameName = linkTable.resolveNameXText(externSheetIndex, externNameIndex, this);if(nameName == null) {return null;}int ix = linkTable.resolveNameXIx(externSheetIndex, externNameIndex);return new ExternalName(nameName, externNameIndex, ix);}
public RegisterUserResult registerUser(RegisterUserRequest request) {request = beforeClientExecution(request);return executeRegisterUser(request);}
public void decode(byte[] blocks, int blocksOffset, int[] values, int valuesOffset, int iterations) {for (int j = 0; j < iterations; ++j) {values[valuesOffset++] = blocks[blocksOffset++] & 0xFF;}}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval arg0) {return fixed(arg0, new NumberEval(2), BoolEval.FALSE, srcRowIndex, srcColumnIndex);}
public final byte[] array() {return protectedArray();}
public int readUByte() {byte[] buf = new byte[1];try {checkEOF(read(buf), 1);} catch (IOException e) {throw new RuntimeException(e);}return LittleEndian.getUByte(buf);}
public static AttrPtg createSkip(int dist) {return new AttrPtg(optiSkip.set(0), dist, null, -1);}
public DescribeUserHierarchyGroupResult describeUserHierarchyGroup(DescribeUserHierarchyGroupRequest request) {request = beforeClientExecution(request);return executeDescribeUserHierarchyGroup(request);}
public User(String path, String userName, String userId, String arn, java.util.Date createDate) {setPath(path);setUserName(userName);setUserId(userId);setArn(arn);setCreateDate(createDate);}
public OpenNLPLemmatizerFilter create(TokenStream in) {try {NLPLemmatizerOp lemmatizerOp = OpenNLPOpsFactory.getLemmatizer(dictionaryFile, lemmatizerModelFile);return new OpenNLPLemmatizerFilter(in, lemmatizerOp);} catch (IOException e) {throw new RuntimeException(e);}}
public void decode(byte[] blocks, int blocksOffset, long[] values, int valuesOffset, int iterations) {for (int i = 0; i < iterations; ++i) {final long byte0 = blocks[blocksOffset++] & 0xFF;final long byte1 = blocks[blocksOffset++] & 0xFF;values[valuesOffset++] = (byte0 << 4) | (byte1 >>> 4);final long byte2 = blocks[blocksOffset++] & 0xFF;values[valuesOffset++] = ((byte1 & 15) << 8) | byte2;}}
public RebootInstanceRequest() {super("HPC", "2016-06-03", "RebootInstance", "hpc");setMethod(MethodType.POST);}
public ListContainerInstancesResult listContainerInstances(ListContainerInstancesRequest request) {request = beforeClientExecution(request);return executeListContainerInstances(request);}
public ListClustersResult listClusters(ListClustersRequest request) {request = beforeClientExecution(request);return executeListClusters(request);}
public static boolean equals(boolean[] array1, boolean[] array2) {if (array1 == array2) {return true;}if (array1 == null || array2 == null || array1.length != array2.length) {return false;}for (int i = 0; i < array1.length; i++) {if (array1[i] != array2[i]) {return false;}}return true;}
public void decode(byte[] blocks, int blocksOffset, int[] values, int valuesOffset, int iterations) {for (int i = 0; i < iterations; ++i) {final int byte0 = blocks[blocksOffset++] & 0xFF;values[valuesOffset++] = byte0 >>> 5;values[valuesOffset++] = (byte0 >>> 2) & 7;final int byte1 = blocks[blocksOffset++] & 0xFF;values[valuesOffset++] = ((byte0 & 3) << 1) | (byte1 >>> 7);values[valuesOffset++] = (byte1 >>> 4) & 7;values[valuesOffset++] = (byte1 >>> 1) & 7;final int byte2 = blocks[blocksOffset++] & 0xFF;values[valuesOffset++] = ((byte1 & 1) << 2) | (byte2 >>> 6);values[valuesOffset++] = (byte2 >>> 3) & 7;values[valuesOffset++] = byte2 & 7;}}
public GetRelationalDatabaseSnapshotResult getRelationalDatabaseSnapshot(GetRelationalDatabaseSnapshotRequest request) {request = beforeClientExecution(request);return executeGetRelationalDatabaseSnapshot(request);}
public int fillFields(byte[] data, int offset, EscherRecordFactory recordFactory) {int bytesRemaining = readHeader( data, offset );int pos = offset + 8;field_1_blipTypeWin32 = data[pos];field_2_blipTypeMacOS = data[pos + 1];System.arraycopy( data, pos + 2, field_3_uid, 0, 16 );field_4_tag = LittleEndian.getShort( data, pos + 18 );field_5_size = LittleEndian.getInt( data, pos + 20 );field_6_ref = LittleEndian.getInt( data, pos + 24 );field_7_offset = LittleEndian.getInt( data, pos + 28 );field_8_usage = data[pos + 32];field_9_name = data[pos + 33];field_10_unused2 = data[pos + 34];field_11_unused3 = data[pos + 35];bytesRemaining -= 36;int bytesRead = 0;if (bytesRemaining > 0) {field_12_blipRecord = (EscherBlipRecord) recordFactory.createRecord( data, pos + 36 );bytesRead = field_12_blipRecord.fillFields( data, pos + 36, recordFactory );}pos += 36 + bytesRead;bytesRemaining -= bytesRead;_remainingData = IOUtils.safelyAllocate(bytesRemaining, MAX_RECORD_LENGTH);System.arraycopy( data, pos, _remainingData, 0, bytesRemaining );return bytesRemaining + 8 + 36 + (field_12_blipRecord == null ? 0 : field_12_blipRecord.getRecordSize()) ;}
@Override public int size() {return size;}
public PhoneNumberValidateResult phoneNumberValidate(PhoneNumberValidateRequest request) {request = beforeClientExecution(request);return executePhoneNumberValidate(request);}
public CreateTransformJobResult createTransformJob(CreateTransformJobRequest request) {request = beforeClientExecution(request);return executeCreateTransformJob(request);}
public synchronized int search(Object o) {final Object[] dumpArray = elementData;final int size = elementCount;if (o != null) {for (int i = size - 1; i >= 0; i--) {if (o.equals(dumpArray[i])) {return size - i;}}} else {for (int i = size - 1; i >= 0; i--) {if (dumpArray[i] == null) {return size - i;}}}return -1;}
public DescribeCacheParametersRequest(String cacheParameterGroupName) {setCacheParameterGroupName(cacheParameterGroupName);}
public void clear() {synchronized (mutex) {delegate().clear();}}
public boolean hasRevSort(RevSort sort) {return sorting.contains(sort);}
public StashListCommand stashList() {return new StashListCommand(repo);}
public PutGroupPolicyRequest(String groupName, String policyName, String policyDocument) {setGroupName(groupName);setPolicyName(policyName);setPolicyDocument(policyDocument);}
public String toString() {return super.get() + "=" + value;}
public void writeByte(int v) {checkPosition(1);_buf[_writeIndex++] = (byte)v;}
public CountryRecord(RecordInputStream in) {field_1_default_country = in.readShort();field_2_current_country = in.readShort();}
public UpdateContainerAgentResult updateContainerAgent(UpdateContainerAgentRequest request) {request = beforeClientExecution(request);return executeUpdateContainerAgent(request);}
public DescribeNodeConfigurationOptionsResult describeNodeConfigurationOptions(DescribeNodeConfigurationOptionsRequest request) {request = beforeClientExecution(request);return executeDescribeNodeConfigurationOptions(request);}
public AddImageRequest() {super("ImageSearch", "2019-03-25", "AddImage", "imagesearch");setUriPattern("/v2/image/add");setMethod(MethodType.POST);}
public BorderFormatting() {field_13_border_styles1    = 0;field_14_border_styles2    = 0;}
public String toFormulaString(String[] operands) {StringBuilder buffer = new StringBuilder();buffer.append(operands[0]);buffer.append(" ");buffer.append(operands[1]);return buffer.toString();}
public ListTagsForStreamResult listTagsForStream(ListTagsForStreamRequest request) {request = beforeClientExecution(request);return executeListTagsForStream(request);}
public HSSFName createName(){NameRecord nameRecord = workbook.createName();HSSFName newName = new HSSFName(this, nameRecord);names.add(newName);return newName;}
public CreateLogPatternResult createLogPattern(CreateLogPatternRequest request) {request = beforeClientExecution(request);return executeCreateLogPattern(request);}
public GetTransitGatewayRouteTablePropagationsResult getTransitGatewayRouteTablePropagations(GetTransitGatewayRouteTablePropagationsRequest request) {request = beforeClientExecution(request);return executeGetTransitGatewayRouteTablePropagations(request);}
public void setup() throws Exception {super.setup();String inputDirProp = getRunData().getConfig().get(ADDINDEXES_INPUT_DIR, null);if (inputDirProp == null) {throw new IllegalArgumentException("config parameter " + ADDINDEXES_INPUT_DIR + " not specified in configuration");}inputDir = FSDirectory.open(Paths.get(inputDirProp));}
public StashDropCommand setAll(boolean all) {this.all = all;return this;}
public ListTrainingJobsForHyperParameterTuningJobResult listTrainingJobsForHyperParameterTuningJob(ListTrainingJobsForHyperParameterTuningJobRequest request) {request = beforeClientExecution(request);return executeListTrainingJobsForHyperParameterTuningJob(request);}
public String toString() {return String.format("Match %s; found %d labels",succeeded() ? "succeeded" : "failed",getLabels().size());}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval arg0) {double result;try {double d = singleOperandEvaluate(arg0, srcRowIndex, srcColumnIndex);result = evaluate(d);checkValue(result);} catch (EvaluationException e) {return e.getErrorEval();}return new NumberEval(result);}
public CacheSecurityGroup authorizeCacheSecurityGroupIngress(AuthorizeCacheSecurityGroupIngressRequest request) {request = beforeClientExecution(request);return executeAuthorizeCacheSecurityGroupIngress(request);}
public String getInflectionType() {return dictionary.getInflectionType(wordId);}
@Override public boolean remove(Object o) {return contains(o) &&(removeValuesForKey(((Multiset.Entry<?>) o).getElement()) > 0);}
public RevCommit next() {RevCommit r = next;next = nextForIterator();return r;}
public BatchAssociateUserStackResult batchAssociateUserStack(BatchAssociateUserStackRequest request) {request = beforeClientExecution(request);return executeBatchAssociateUserStack(request);}
public ScenarioProtectRecord clone() {return copy();}
public final Class getBundleClass() {return bundleClass;}
public void nextBuffer() {if (1+bufferUpto == buffers.length) {int[][] newBuffers = new int[(int) (buffers.length*1.5)][];System.arraycopy(buffers, 0, newBuffers, 0, buffers.length);buffers = newBuffers;}buffer = buffers[1+bufferUpto] = allocator.getIntBlock();bufferUpto++;intUpto = 0;intOffset += INT_BLOCK_SIZE;}
public DeleteVpnGatewayRequest(String vpnGatewayId) {setVpnGatewayId(vpnGatewayId);}
public static Encoder getEncoder(Format format, int version, int bitsPerValue) {checkVersion(version);return BulkOperation.of(format, bitsPerValue);}
public ClassificationResult(T assignedClass, double score) {this.assignedClass = assignedClass;this.score = score;}
public CreateRelationalDatabaseSnapshotResult createRelationalDatabaseSnapshot(CreateRelationalDatabaseSnapshotRequest request) {request = beforeClientExecution(request);return executeCreateRelationalDatabaseSnapshot(request);}
public NameRecord addName(NameRecord name) {getOrCreateLinkTable().addName(name);return name;}
public void serialize(LittleEndianOutput out) {out.writeShort(getFirstRow());out.writeShort(getLastRow());out.writeByte(getFirstColumn());out.writeByte(getLastColumn());}
public String getKey() {return key;}
public GetBlockPublicAccessConfigurationResult getBlockPublicAccessConfiguration(GetBlockPublicAccessConfigurationRequest request) {request = beforeClientExecution(request);return executeGetBlockPublicAccessConfiguration(request);}
public static long getResultSize(byte[] delta) {int p = 0;int c;do {c = delta[p++] & 0xff;} while ((c & 0x80) != 0);long resLen = 0;int shift = 0;do {c = delta[p++] & 0xff;resLen |= ((long) (c & 0x7f)) << shift;shift += 7;} while ((c & 0x80) != 0);return resLen;}
public long ramBytesUsed() {return RamUsageEstimator.alignObjectSize(RamUsageEstimator.NUM_BYTES_OBJECT_HEADER + Integer.BYTES);}
public NoteRecord() {field_6_author = "";field_3_flags = 0;field_7_padding = DEFAULT_PADDING; }
public CellReference[] getAllReferencedCells() {if(_isSingleCell) {return  new CellReference[] { _firstCell, };}int minRow = Math.min(_firstCell.getRow(), _lastCell.getRow());int maxRow = Math.max(_firstCell.getRow(), _lastCell.getRow());int minCol = Math.min(_firstCell.getCol(), _lastCell.getCol());int maxCol = Math.max(_firstCell.getCol(), _lastCell.getCol());String sheetName = _firstCell.getSheetName();List<CellReference> refs = new ArrayList<>();for(int row=minRow; row<=maxRow; row++) {for(int col=minCol; col<=maxCol; col++) {CellReference ref = new CellReference(sheetName, row, col, _firstCell.isRowAbsolute(), _firstCell.isColAbsolute());refs.add(ref);}}return refs.toArray(new CellReference[0]);}
public String[] listAll() {ensureOpen();String[] res = entries.keySet().toArray(new String[entries.size()]);for (int i = 0; i < res.length; i++) {res[i] = segmentName + res[i];}return res;}
public UpdateDataRetentionResult updateDataRetention(UpdateDataRetentionRequest request) {request = beforeClientExecution(request);return executeUpdateDataRetention(request);}
public CreateDistributionRequest(DistributionConfig distributionConfig) {setDistributionConfig(distributionConfig);}
public DescribeBatchPredictionsResult describeBatchPredictions(DescribeBatchPredictionsRequest request) {request = beforeClientExecution(request);return executeDescribeBatchPredictions(request);}
public float getScore(int index) {return scores[index];}
public BatchUpdatePhoneNumberResult batchUpdatePhoneNumber(BatchUpdatePhoneNumberRequest request) {request = beforeClientExecution(request);return executeBatchUpdatePhoneNumber(request);}
public LMSimilarity(CollectionModel collectionModel) {this.collectionModel = collectionModel;}
public GetGlobalSettingsResult getGlobalSettings(GetGlobalSettingsRequest request) {request = beforeClientExecution(request);return executeGetGlobalSettings(request);}
public CreateHITTypeResult createHITType(CreateHITTypeRequest request) {request = beforeClientExecution(request);return executeCreateHITType(request);}
public MLTConfig build() {return new MLTConfig(this);}
public CharsRef(String string) {this.chars = string.toCharArray();this.offset = 0;this.length = chars.length;}
public ListFargateProfilesResult listFargateProfiles(ListFargateProfilesRequest request) {request = beforeClientExecution(request);return executeListFargateProfiles(request);}
public Entry<K, V> floorEntry(K key) {return immutableCopy(findBounded(key, FLOOR));}
public boolean equals( Object o ) {return o instanceof NorwegianStemmer;}
public DeleteVaultNotificationsResult deleteVaultNotifications(DeleteVaultNotificationsRequest request) {request = beforeClientExecution(request);return executeDeleteVaultNotifications(request);}
public static boolean endsWith(char s[], int len, String suffix) {final int suffixLen = suffix.length();if (suffixLen > len)return false;for (int i = suffixLen - 1; i >= 0; i--)if (s[len -(suffixLen - i)] != suffix.charAt(i))return false;return true;}
public synchronized void setRequireDimCount(String dimName, boolean v) {DimConfig ft = fieldTypes.get(dimName);if (ft == null) {ft = new DimConfig();fieldTypes.put(dimName, ft);}ft.requireDimCount = v;}
public HSSFName getName(String name) {int nameIndex = getNameIndex(name);if (nameIndex < 0) {return null;}return names.get(nameIndex);}
public ScriptBootstrapActionConfig(String path, java.util.List<String> args) {setPath(path);setArgs(args);}
public RegisterApplicationRevisionResult registerApplicationRevision(RegisterApplicationRevisionRequest request) {request = beforeClientExecution(request);return executeRegisterApplicationRevision(request);}
public SendTestEventNotificationResult sendTestEventNotification(SendTestEventNotificationRequest request) {request = beforeClientExecution(request);return executeSendTestEventNotification(request);}
public void setRefLogIdent(PersonIdent pi) {refLogIdent = pi;}
public GetDomainDeliverabilityCampaignResult getDomainDeliverabilityCampaign(GetDomainDeliverabilityCampaignRequest request) {request = beforeClientExecution(request);return executeGetDomainDeliverabilityCampaign(request);}
public String toFormulaString() {StringBuilder b = new StringBuilder();b.append("{");for (int y = 0; y < _nRows; y++) {if (y > 0) {b.append(";");}for (int x = 0; x < _nColumns; x++) {if (x > 0) {b.append(",");}Object o = _arrayValues[getValueIndex(x, y)];b.append(getConstantText(o));}}b.append("}");return b.toString();}
public ShingleFilterFactory(Map<String, String> args) {super(args);maxShingleSize = getInt(args, "maxShingleSize", ShingleFilter.DEFAULT_MAX_SHINGLE_SIZE);if (maxShingleSize < 2) {throw new IllegalArgumentException("Invalid maxShingleSize (" + maxShingleSize + ") - must be at least 2");}minShingleSize = getInt(args, "minShingleSize", ShingleFilter.DEFAULT_MIN_SHINGLE_SIZE);if (minShingleSize < 2) {throw new IllegalArgumentException("Invalid minShingleSize (" + minShingleSize + ") - must be at least 2");}if (minShingleSize > maxShingleSize) {throw new IllegalArgumentException("Invalid minShingleSize (" + minShingleSize + ") - must be no greater than maxShingleSize (" + maxShingleSize + ")");}outputUnigrams = getBoolean(args, "outputUnigrams", true);outputUnigramsIfNoShingles = getBoolean(args, "outputUnigramsIfNoShingles", false);tokenSeparator = get(args, "tokenSeparator", ShingleFilter.DEFAULT_TOKEN_SEPARATOR);fillerToken = get(args, "fillerToken", ShingleFilter.DEFAULT_FILLER_TOKEN);if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public UpdateRelationalDatabaseParametersResult updateRelationalDatabaseParameters(UpdateRelationalDatabaseParametersRequest request) {request = beforeClientExecution(request);return executeUpdateRelationalDatabaseParameters(request);}
public static Collection<ParseTree> findAllRuleNodes(ParseTree t, int ruleIndex) {return findAllNodes(t, ruleIndex, false);}
public int getObjectCount() {return entryCount;}
public ActionTransition(ATNState target, int ruleIndex, int actionIndex, boolean isCtxDependent) {super(target);this.ruleIndex = ruleIndex;this.actionIndex = actionIndex;this.isCtxDependent = isCtxDependent;}
public long get(int index) {final int blockOffset = index / valuesPerBlock;final long skip = ((long) blockOffset) << 3;try {in.seek(startPointer + skip);long block = in.readLong();final int offsetInBlock = index % valuesPerBlock;return (block >>> (offsetInBlock * bitsPerValue)) & mask;} catch (IOException e) {throw new IllegalStateException("failed", e);}}
public String getSignerType() {return "BEARERTOKEN";}
public PipedOutputStream(PipedInputStream target) throws IOException {connect(target);}
public DeleteLedgerResult deleteLedger(DeleteLedgerRequest request) {request = beforeClientExecution(request);return executeDeleteLedger(request);}
public GetCognitoEventsResult getCognitoEvents(GetCognitoEventsRequest request) {request = beforeClientExecution(request);return executeGetCognitoEvents(request);}
public NameXPtg getNameXPtg(String name, SheetIdentifier sheet) {int sheetRefIndex = getSheetExtIx(sheet);return _iBook.getNameXPtg(name, sheetRefIndex, _uBook.getUDFFinder());}
public ListResolverEndpointsResult listResolverEndpoints(ListResolverEndpointsRequest request) {request = beforeClientExecution(request);return executeListResolverEndpoints(request);}
public String readLine() {try {return reader.readLine();} catch (IOException e) {throw new IOError(e);}}
public int hash2(char carray[]) {int hash = 5381;for (int i = 0; i < carray.length; i++) {char d = carray[i];hash = ((hash << 5) + hash) + d & 0x00FF;hash = ((hash << 5) + hash) + d >> 8;}return hash;}
public static long toBookSheetColumn(int bookIndex, int sheetIndex, int columnIndex) {return ((bookIndex   & 0xFFFFL) << 48)  +((sheetIndex  & 0xFFFFL) << 32) +((columnIndex & 0xFFFFL) << 0);}
public CreateConfigurationProfileResult createConfigurationProfile(CreateConfigurationProfileRequest request) {request = beforeClientExecution(request);return executeCreateConfigurationProfile(request);}
public ReplicationGroup startMigration(StartMigrationRequest request) {request = beforeClientExecution(request);return executeStartMigration(request);}
public OffsetLimitTokenFilter(TokenStream input, int offsetLimit) {super(input);this.offsetLimit = offsetLimit;}
public final void write(byte[] b, int off, int len)throws IOException {while (0 < len) {final int n = Math.min(len, BYTES_TO_WRITE_BEFORE_CANCEL_CHECK);count += n;if (checkCancelAt <= count) {if (writeMonitor.isCancelled()) {throw new IOException(JGitText.get().packingCancelledDuringObjectsWriting);}checkCancelAt = count + BYTES_TO_WRITE_BEFORE_CANCEL_CHECK;}out.write(b, off, n);md.update(b, off, n);off += n;len -= n;}}
public Cell merge(Cell m, Cell e) {Cell n = new Cell();if (m.skip != e.skip) {return null;}if (m.cmd >= 0) {if (e.cmd >= 0) {if (m.cmd == e.cmd) {n.cmd = m.cmd;} else {return null;}} else {n.cmd = m.cmd;}} else {n.cmd = e.cmd;}if (m.ref >= 0) {if (e.ref >= 0) {if (m.ref == e.ref) {if (m.skip == e.skip) {n.ref = m.ref;} else {return null;}} else {return null;}} else {n.ref = m.ref;}} else {n.ref = e.ref;}n.cnt = m.cnt + e.cnt;n.skip = m.skip;return n;}
public GetCampaignActivitiesResult getCampaignActivities(GetCampaignActivitiesRequest request) {request = beforeClientExecution(request);return executeGetCampaignActivities(request);}
public long estimateBytesUsed() {return bytesUsed;}
public FunctionNameEval(String functionName) {_functionName = functionName;}
public final float averageBytesPerChar() {return averageBytesPerChar;}
public CreateCacheSecurityGroupRequest(String cacheSecurityGroupName, String description) {setCacheSecurityGroupName(cacheSecurityGroupName);setDescription(description);}
public void removeAt(int index) {System.arraycopy(mKeys, index + 1, mKeys, index, mSize - (index + 1));System.arraycopy(mValues, index + 1, mValues, index, mSize - (index + 1));mSize--;}
public DescribeIndexFieldsResult describeIndexFields(DescribeIndexFieldsRequest request) {request = beforeClientExecution(request);return executeDescribeIndexFields(request);}
public void remove(int key) {delete(key);}
public ShortBuffer duplicate() {ByteBuffer bb = byteBuffer.duplicate().order(byteBuffer.order());ShortToByteBufferAdapter buf = new ShortToByteBufferAdapter(bb);buf.limit = limit;buf.position = position;buf.mark = mark;return buf;}
public void addDbcell(int cell){if (field_5_dbcells == null){field_5_dbcells = new IntList();}field_5_dbcells.add(cell);}
public DeleteSubnetResult deleteSubnet(DeleteSubnetRequest request) {request = beforeClientExecution(request);return executeDeleteSubnet(request);}
public List<HSSFPictureData> getAllPictures(){List<HSSFPictureData> pictures = new ArrayList<>();for (org.apache.poi.hssf.record.Record r : workbook.getRecords()) {if (r instanceof AbstractEscherHolderRecord) {((AbstractEscherHolderRecord) r).decode();List<EscherRecord> escherRecords = ((AbstractEscherHolderRecord) r).getEscherRecords();searchForPictures(escherRecords, pictures);}}return Collections.unmodifiableList(pictures);}
public DescribeWorkspacesConnectionStatusResult describeWorkspacesConnectionStatus(DescribeWorkspacesConnectionStatusRequest request) {request = beforeClientExecution(request);return executeDescribeWorkspacesConnectionStatus(request);}
public String toString() {return "MultiDocsAndPositionsEnum(" + Arrays.toString(getSubs()) + ")";}
public InvokeServiceAsyncRequest() {super("industry-brain", "2018-07-12", "InvokeServiceAsync");setMethod(MethodType.POST);}
public AuthorizeSecurityGroupIngressRequest(String groupName, java.util.List<IpPermission> ipPermissions) {setGroupName(groupName);setIpPermissions(ipPermissions);}
public static byte[] readData(InputStream stream, String section ) throws IOException {try {StringBuilder sectionText = new StringBuilder();boolean inSection = false;int c = stream.read();while ( c != -1 ) {switch ( c ) {case '[':inSection = true;break;case '\n':case '\r':inSection = false;sectionText = new StringBuilder();break;case ']':inSection = false;if ( sectionText.toString().equals( section ) ) return readData( stream, '[' );sectionText = new StringBuilder();break;default:if ( inSection ) sectionText.append( (char) c );}c = stream.read();}} finally {stream.close();}throw new IOException( "Section '" + section + "' not found" );}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval numberVE) {int number;try {number = OperandResolver.coerceValueToInt(numberVE);} catch (EvaluationException e) {return ErrorEval.VALUE_INVALID;}if (number < 0) {return ErrorEval.NUM_ERROR;}return new NumberEval(factorial(number).longValue());}
public final LexerActionExecutor getLexerActionExecutor() {return lexerActionExecutor;}
public EnableUserResult enableUser(EnableUserRequest request) {request = beforeClientExecution(request);return executeEnableUser(request);}
public void fillSlice(BytesRef b, long start, int length) {assert length >= 0: "length=" + length;assert length <= blockSize+1: "length=" + length;b.length = length;if (length == 0) {return;}final int index = (int) (start >> blockBits);final int offset = (int) (start & blockMask);if (blockSize - offset >= length) {b.bytes = blocks[index];b.offset = offset;} else {b.bytes = new byte[length];b.offset = 0;System.arraycopy(blocks[index], offset, b.bytes, 0, blockSize-offset);System.arraycopy(blocks[1+index], 0, b.bytes, blockSize-offset, length-(blockSize-offset));}}
public DescribeJournalS3ExportResult describeJournalS3Export(DescribeJournalS3ExportRequest request) {request = beforeClientExecution(request);return executeDescribeJournalS3Export(request);}
public void setCoordinates(int x1, int y1, int x2, int y2) {_spgrRecord.setRectY1(y1);_spgrRecord.setRectY2(y2);_spgrRecord.setRectX1(x1);_spgrRecord.setRectX2(x2);}
public DescribeTagsResult describeTags(DescribeTagsRequest request) {request = beforeClientExecution(request);return executeDescribeTags(request);}
public int doLogic() {return 1;}
public DeleteCustomerGatewayResult deleteCustomerGateway(DeleteCustomerGatewayRequest request) {request = beforeClientExecution(request);return executeDeleteCustomerGateway(request);}
public static Map newContext(IndexSearcher searcher) {Map context = new IdentityHashMap();context.put("searcher", searcher);return context;}
public NameRecord getSpecificBuiltinRecord(byte builtInCode, int sheetNumber) {Iterator<NameRecord> iterator = _definedNames.iterator();while (iterator.hasNext()) {NameRecord record = iterator.next();if (record.getBuiltInName() == builtInCode && record.getSheetNumber() == sheetNumber) {return record;}}return null;}
public final double readDouble() throws IOException {return Double.longBitsToDouble(readLong());}
public void write(byte[] buffer, int offset, int count) throws IOException {super.write(buffer, offset, count);}
public TokenStream create(TokenStream input) {return new PersianNormalizationFilter(input);}
public SpanishLightStemFilterFactory(Map<String,String> args) {super(args);if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public SmallDocSet(int size) {intSet = new SentinelIntSet(size, -1);}
public RawCharSequence(byte[] buf, int start, int end) {buffer = buf;startPtr = start;endPtr = end;}
public GetCustomVerificationEmailTemplateResult getCustomVerificationEmailTemplate(GetCustomVerificationEmailTemplateRequest request) {request = beforeClientExecution(request);return executeGetCustomVerificationEmailTemplate(request);}
public SendMessageBatchRequest(String queueUrl, java.util.List<SendMessageBatchRequestEntry> entries) {setQueueUrl(queueUrl);setEntries(entries);}
public void writeInt(int v) {writeContinueIfRequired(4);_ulrOutput.writeInt(v);}
public DescribeDataSourcesResult describeDataSources(DescribeDataSourcesRequest request) {request = beforeClientExecution(request);return executeDescribeDataSources(request);}
public ListRoomsResult listRooms(ListRoomsRequest request) {request = beforeClientExecution(request);return executeListRooms(request);}
public char getConversion() {return c;}
public boolean equals(Object _other) {FieldAndTerm other = (FieldAndTerm) _other;return other.field.equals(field) && term.bytesEquals(other.term);}
public CreateConfigurationSetEventDestinationResult createConfigurationSetEventDestination(CreateConfigurationSetEventDestinationRequest request) {request = beforeClientExecution(request);return executeCreateConfigurationSetEventDestination(request);}
public Ole10Native(String label, String filename, String command, byte[] data) {setLabel(label);setFileName(filename);setCommand(command);setDataBuffer(data);mode = EncodingMode.parsed;}
public String toString() {StringBuilder sb = new StringBuilder();if (fetchResult != null)sb.append(fetchResult.toString());elsesb.append("No fetch result");sb.append("\n");if (mergeResult != null)sb.append(mergeResult.toString());else if (rebaseResult != null)sb.append(rebaseResult.toString());elsesb.append("No update result");return sb.toString();}
public static Cell createCell(Row row, int column, String value) {return createCell(row, column, value, null);}
public TokenStream create(TokenStream input) {return new HindiNormalizationFilter(input);}
public DescribeAddressesResult describeAddresses() {return describeAddresses(new DescribeAddressesRequest());}
public SimpleQQParser(String qqName, String indexField) {this(new String[] { qqName }, indexField);}
public void dispatch(RefsChangedListener listener) {listener.onRefsChanged(this);}
public SnowballFilter(TokenStream in, String name) {super(in);try {Class<? extends SnowballStemmer> stemClass =Class.forName("org.tartarus.snowball.ext." + name + "Stemmer").asSubclass(SnowballStemmer.class);stemmer = stemClass.getConstructor().newInstance();} catch (Exception e) {
public UpgradeAppliedSchemaResult upgradeAppliedSchema(UpgradeAppliedSchemaRequest request) {request = beforeClientExecution(request);return executeUpgradeAppliedSchema(request);}
public String getParent() {int length = path.length(), firstInPath = 0;if (separatorChar == '\\' && length > 2 && path.charAt(1) == ':') {firstInPath = 2;}int index = path.lastIndexOf(separatorChar);if (index == -1 && firstInPath > 0) {index = 2;}if (index == -1 || path.charAt(length - 1) == separatorChar) {return null;}if (path.indexOf(separatorChar) == index&& path.charAt(firstInPath) == separatorChar) {return path.substring(0, index + 1);}return path.substring(0, index);}
public BufferedChecksumIndexInput(IndexInput main) {super("BufferedChecksumIndexInput(" + main + ")");this.main = main;this.digest = new BufferedChecksum(new CRC32());}
public final void remove(RevFlagSet set) {flags &= ~set.mask;}
public boolean equals(Object obj) {if (this == obj)return true;if (obj == null)return false;if (getClass() != obj.getClass())return false;return true;}
public GetFaceSearchResult getFaceSearch(GetFaceSearchRequest request) {request = beforeClientExecution(request);return executeGetFaceSearch(request);}
public DescribeUserStackAssociationsResult describeUserStackAssociations(DescribeUserStackAssociationsRequest request) {request = beforeClientExecution(request);return executeDescribeUserStackAssociations(request);}
public void close() throws IOException {in.close();in = new ClosedInputStream();}
public CreateBranchCommand branchCreate() {return new CreateBranchCommand(repo);}
public void serialize(LittleEndianOutput out) {out.writeShort(rt);out.writeShort(grbitFrt);out.writeShort(wOffset);out.writeShort(at);out.writeShort(grbit);if(unused != null)out.writeShort(unused);}
public StringBuilder insert(int offset, Object obj) {insert0(offset, obj == null ? "null" : obj.toString());return this;}
public int next() {int res = child;if (child != TaxonomyReader.INVALID_ORDINAL) {child = siblings[child];}return res;}
public DeleteStackResult deleteStack(DeleteStackRequest request) {request = beforeClientExecution(request);return executeDeleteStack(request);}
public NorwegianMinimalStemFilterFactory(Map<String,String> args) {super(args);String variant = get(args, "variant");if (variant == null || "nb".equals(variant)) {flags = BOKMAAL;} else if ("nn".equals(variant)) {flags = NYNORSK;} else if ("no".equals(variant)) {flags = BOKMAAL | NYNORSK;} else {throw new IllegalArgumentException("invalid variant: " + variant);}if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public String toString() {return "Z(" + z + ")";}
public static org.apache.poi.hssf.record.Record create(RecordInputStream in) {switch (in.remaining()) {case 0:return instance;case 2:return new InterfaceHdrRecord(in);}throw new RecordFormatException("Invalid record data size: " + in.remaining());}
public int getCellsPnt() {int size = 0;for (Row row : rows)size += row.getCellsPnt();return size;}
public boolean equals(Object obj) {if (obj == this) {return true;}else if (!(obj instanceof LexerActionExecutor)) {return false;}LexerActionExecutor other = (LexerActionExecutor)obj;return hashCode == other.hashCode&& Arrays.equals(lexerActions, other.lexerActions);}
public static final Analyzer createAnalyzer(String className) throws Exception{final Class<? extends Analyzer> clazz = Class.forName(className).asSubclass(Analyzer.class);try {Constructor<? extends Analyzer> cnstr = clazz.getConstructor(Version.class);return cnstr.newInstance(Version.LATEST);} catch (NoSuchMethodException nsme) {return clazz.getConstructor().newInstance();}}
public GetSegmentVersionsResult getSegmentVersions(GetSegmentVersionsRequest request) {request = beforeClientExecution(request);return executeGetSegmentVersions(request);}
public int getDeltaBaseCacheLimit() {return deltaBaseCacheLimit;}
public GroupMerger(Sort groupSort) {groupComp = new GroupComparator<>(groupSort);queue = new TreeSet<>(groupComp);groupsSeen = new HashMap<>();}
public long get(int index) {final int o = index >>> 4;final int b = index & 15;final int shift = b << 2;return (blocks[o] >>> shift) & 15L;}
public FileIdCluster( int drawingGroupId, int numShapeIdsUsed ) {this.field_1_drawingGroupId = drawingGroupId;this.field_2_numShapeIdsUsed = numShapeIdsUsed;}
public CharArrayIterator clone() {CharArrayIterator clone = new CharArrayIterator();clone.setText(array, start, length);clone.index = index;return clone;}
public DescribeReservedNodesResult describeReservedNodes(DescribeReservedNodesRequest request) {request = beforeClientExecution(request);return executeDescribeReservedNodes(request);}
public ObjectWalk(Repository repo, int depth) {super(repo);this.depth = depth;this.deepenNots = Collections.emptyList();this.UNSHALLOW = newFlag("UNSHALLOW"); this.REINTERESTING = newFlag("REINTERESTING"); this.DEEPEN_NOT = newFlag("DEEPEN_NOT"); }
public boolean isRefLogDisabled() {return refLogMessage == null;}
public SetLoadBalancerListenerSSLCertificateResult setLoadBalancerListenerSSLCertificate(SetLoadBalancerListenerSSLCertificateRequest request) {request = beforeClientExecution(request);return executeSetLoadBalancerListenerSSLCertificate(request);}
public DescribeRulesPackagesResult describeRulesPackages(DescribeRulesPackagesRequest request) {request = beforeClientExecution(request);return executeDescribeRulesPackages(request);}
public byte readByte() throws IOException {return primitiveTypes.readByte();}
public String getConversion() {return s;}
public StandardSyntaxParserTokenManager(CharStream stream, int lexState){this(stream);SwitchTo(lexState);}
public TokenStream create(TokenStream input) {return new TurkishLowerCaseFilter(input);}
public String toString() {return "B";}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval arg0) {return evaluate(srcRowIndex, srcColumnIndex, arg0, DEFAULT_ARG1);}
public void doubleField(FieldInfo fieldInfo, double value) {doc.add(new StoredField(fieldInfo.name, value));}
public GetDistributionConfigRequest(String id) {setId(id);}
public DescribeCacheSecurityGroupsResult describeCacheSecurityGroups() {return describeCacheSecurityGroups(new DescribeCacheSecurityGroupsRequest());}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval arg0) {double d;try {ValueEval ve = OperandResolver.getSingleValue(arg0, srcRowIndex, srcColumnIndex);d = OperandResolver.coerceValueToDouble(ve);} catch (EvaluationException e) {return e.getErrorEval();}if (d == 0.0) { return NumberEval.ZERO;}return new NumberEval(d / 100);}
public boolean containsCell(int rowIndex, int columnIndex) {if (columnIndex < _firstColumnIndex) {return false;}if (columnIndex > _lastColumnIndex) {return false;}if (rowIndex < _firstRowIndex) {return false;}if (rowIndex > _lastRowIndex) {return false;}return true;}
public GetSegmentVersionResult getSegmentVersion(GetSegmentVersionRequest request) {request = beforeClientExecution(request);return executeGetSegmentVersion(request);}
public final FloatBuffer put(float[] src, int srcOffset, int byteCount) {throw new ReadOnlyBufferException();}
public final IntBuffer put(int[] src) {return put(src, 0, src.length);}
public SearchFaceRequest() {super("LinkFace", "2018-07-20", "SearchFace");setProtocol(ProtocolType.HTTPS);setMethod(MethodType.POST);}
public TagStreamResult tagStream(TagStreamRequest request) {request = beforeClientExecution(request);return executeTagStream(request);}
public String getAccessKeyId() {return this.accessKeyId;}
public ET previous() {if (expectedModCount == list.modCount) {if (link != list.voidLink) {lastLink = link;link = link.previous;pos--;return lastLink.data;}throw new NoSuchElementException();}throw new ConcurrentModificationException();}
public CreateLBCookieStickinessPolicyResult createLBCookieStickinessPolicy(CreateLBCookieStickinessPolicyRequest request) {request = beforeClientExecution(request);return executeCreateLBCookieStickinessPolicy(request);}
public CreateDataSourceFromRDSResult createDataSourceFromRDS(CreateDataSourceFromRDSRequest request) {request = beforeClientExecution(request);return executeCreateDataSourceFromRDS(request);}
public CreateReceiptFilterResult createReceiptFilter(CreateReceiptFilterRequest request) {request = beforeClientExecution(request);return executeCreateReceiptFilter(request);}
public final byte get(int index) {checkIndex(index);return backingArray[offset + index];}
public CherryPickCommand include(AnyObjectId commit) {return include(commit.getName(), commit);}
public ATNDeserializationOptions() {this.verifyATN = true;this.generateRuleBypassTransitions = false;}
public ListIdentityPoliciesResult listIdentityPolicies(ListIdentityPoliciesRequest request) {request = beforeClientExecution(request);return executeListIdentityPolicies(request);}
public static boolean isValidCode(int errorCode) {for (FormulaError error : values()) {if (error.getCode() == errorCode) return true;if (error.getLongCode() == errorCode) return true;}return false;}
public RKRecord(RecordInputStream in) {super(in);field_4_rk_number = in.readInt();}
public void copyTo(ByteBuffer b) {b.put(toHexByteArray());}
public String toString(){StringBuilder buffer = new StringBuilder();buffer.append("[DAT]\n");buffer.append("    .options              = ").append("0x").append(HexDump.toHex(  getOptions ())).append(" (").append( getOptions() ).append(" )");buffer.append(System.getProperty("line.separator"));buffer.append("         .horizontalBorder         = ").append(isHorizontalBorder()).append('\n');buffer.append("         .verticalBorder           = ").append(isVerticalBorder()).append('\n');buffer.append("         .border                   = ").append(isBorder()).append('\n');buffer.append("         .showSeriesKey            = ").append(isShowSeriesKey()).append('\n');buffer.append("[/DAT]\n");return buffer.toString();}
public UpdateDashboardResult updateDashboard(UpdateDashboardRequest request) {request = beforeClientExecution(request);return executeUpdateDashboard(request);}
public RegisterTagRequest() {super("CloudPhoto", "2017-07-11", "RegisterTag", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public DiffCommand setPathFilter(TreeFilter pathFilter) {this.pathFilter = pathFilter;return this;}
public boolean markSupported() {return true;}
public String toString() {StringBuilder sb = new StringBuilder(getClass().getSimpleName() + ": ");sb.append("maxThreadCount=").append(maxThreadCount).append(", ");sb.append("maxMergeCount=").append(maxMergeCount).append(", ");sb.append("ioThrottle=").append(doAutoIOThrottle);return sb.toString();}
public synchronized void println(String str) {print(str);newline();}
public UpdateApiResult updateApi(UpdateApiRequest request) {request = beforeClientExecution(request);return executeUpdateApi(request);}
public FlushStageAuthorizersCacheResult flushStageAuthorizersCache(FlushStageAuthorizersCacheRequest request) {request = beforeClientExecution(request);return executeFlushStageAuthorizersCache(request);}
public BasicQueryFactory(int maxBasicQueries) {this.maxBasicQueries = maxBasicQueries;this.queriesMade = 0;}
public TrackingRefUpdate getTrackingRefUpdate(String localName) {return updates.get(localName);}
public String toString() {StringBuilder buffer = new StringBuilder();buffer.append("[CATLAB]\n");buffer.append("    .rt      =").append(HexDump.shortToHex(rt)).append('\n');buffer.append("    .grbitFrt=").append(HexDump.shortToHex(grbitFrt)).append('\n');buffer.append("    .wOffset =").append(HexDump.shortToHex(wOffset)).append('\n');buffer.append("    .at      =").append(HexDump.shortToHex(at)).append('\n');buffer.append("    .grbit   =").append(HexDump.shortToHex(grbit)).append('\n');if(unused != null)buffer.append("    .unused  =").append(HexDump.shortToHex(unused)).append('\n');buffer.append("[/CATLAB]\n");return buffer.toString();}
public EnableDirectoryResult enableDirectory(EnableDirectoryRequest request) {request = beforeClientExecution(request);return executeEnableDirectory(request);}
public IntBuffer put(int[] src, int srcOffset, int intCount) {if (intCount > remaining()) {throw new BufferOverflowException();}System.arraycopy(src, srcOffset, backingArray, offset + position, intCount);position += intCount;return this;}
public String toString() {StringBuilder buffer = new StringBuilder();buffer.append("[PROT4REVPASSWORD]\n");buffer.append("    .password = ").append(HexDump.shortToHex(field_1_password)).append("\n");buffer.append("[/PROT4REVPASSWORD]\n");return buffer.toString();}
public DescribeProjectVersionsResult describeProjectVersions(DescribeProjectVersionsRequest request) {request = beforeClientExecution(request);return executeDescribeProjectVersions(request);}
public UpdateHostedZoneCommentResult updateHostedZoneComment(UpdateHostedZoneCommentRequest request) {request = beforeClientExecution(request);return executeUpdateHostedZoneComment(request);}
public Rescorer getRescorer(Bindings bindings) {return new ExpressionRescorer(this, bindings);}
public SortedSet<E> headSet(E end) {return headSet(end, false);}
final public QueryNode DisjQuery(CharSequence field) throws ParseException {QueryNode first, c;Vector<QueryNode> clauses = null;first = ConjQuery(field);label_2:while (true) {switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {case OR:;break;default:jj_la1[3] = jj_gen;break label_2;}jj_consume_token(OR);c = ConjQuery(field);if (clauses == null) {clauses = new Vector<QueryNode>();clauses.addElement(first);}clauses.addElement(c);}if (clauses != null) {{if (true) return new OrQueryNode(clauses);}} else {{if (true) return first;}}throw new Error("Missing return statement in function");}
public DataValidationConstraint createExplicitListConstraint(String[] listOfValues) {return DVConstraint.createExplicitListConstraint(listOfValues);}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval arg0,ValueEval arg1) {String s0;String s1;try {s0 = evaluateStringArg(arg0, srcRowIndex, srcColumnIndex);s1 = evaluateStringArg(arg1, srcRowIndex, srcColumnIndex);} catch (EvaluationException e) {return e.getErrorEval();}return BoolEval.valueOf(s0.equals(s1));}
public boolean offer(E o) {return addLastImpl(o);}
public ListInvalidationsRequest(String distributionId) {setDistributionId(distributionId);}
public TagPhotoRequest() {super("CloudPhoto", "2017-07-11", "TagPhoto", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public CreateFleetResult createFleet(CreateFleetRequest request) {request = beforeClientExecution(request);return executeCreateFleet(request);}
public GetTransitGatewayAttachmentPropagationsResult getTransitGatewayAttachmentPropagations(GetTransitGatewayAttachmentPropagationsRequest request) {request = beforeClientExecution(request);return executeGetTransitGatewayAttachmentPropagations(request);}
public ListWorkteamsResult listWorkteams(ListWorkteamsRequest request) {request = beforeClientExecution(request);return executeListWorkteams(request);}
public DetachVpnGatewayResult detachVpnGateway(DetachVpnGatewayRequest request) {request = beforeClientExecution(request);return executeDetachVpnGateway(request);}
public ListGeoLocationsResult listGeoLocations() {return listGeoLocations(new ListGeoLocationsRequest());}
public String toString() {return getClass().getName() + " [" +getStringValue() +"]";}
public static double decodeNumber(int number) {long raw_number = number;raw_number = raw_number >> 2;double rvalue = 0;if ((number & 0x02) == 0x02){rvalue = raw_number;}else{rvalue = Double.longBitsToDouble(raw_number << 34);}if ((number & 0x01) == 0x01){rvalue /= 100;}return rvalue;}
public long get(long index) {assert index >= 0 && index < valueCount;final int block = (int) (index >>> blockShift);final int idx = (int) (index & blockMask);return (minValues == null ? 0 : minValues[block]) + subReaders[block].get(idx);}
public UpdatePublishingDestinationResult updatePublishingDestination(UpdatePublishingDestinationRequest request) {request = beforeClientExecution(request);return executeUpdatePublishingDestination(request);}
public void notifyDeleteCell(EvaluationCell cell) {int sheetIndex = getSheetIndex(cell.getSheet());_cache.notifyDeleteCell(_workbookIx, sheetIndex, cell);}
public Request<GetPolicyRequest> marshall(GetPolicyRequest getPolicyRequest) {if (getPolicyRequest == null) {throw new SdkClientException("Invalid argument passed to marshall(...)");}Request<GetPolicyRequest> request = new DefaultRequest<GetPolicyRequest>(getPolicyRequest, "AmazonIdentityManagement");request.addParameter("Action", "GetPolicy");request.addParameter("Version", "2010-05-08");request.setHttpMethod(HttpMethodName.POST);if (getPolicyRequest.getPolicyArn() != null) {request.addParameter("PolicyArn", StringUtils.fromString(getPolicyRequest.getPolicyArn()));}return request;}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval real_num, ValueEval i_num) {return this.evaluate(srcRowIndex, srcColumnIndex, real_num, i_num, new StringEval(DEFAULT_SUFFIX));}
public int fillFields(byte[] data, int offset, EscherRecordFactory recordFactory) { readHeader( data, offset );int pos            = offset + 8;int size           = 0;field_1_numShapes   =  LittleEndian.getInt( data, pos + size );     size += 4;field_2_lastMSOSPID =  LittleEndian.getInt( data, pos + size );     size += 4;return getRecordSize();}
public final CharsetEncoder reset() {status = INIT;implReset();return this;}
public void emit(Token token) {this._token = token;}
public AbstractTreeIterator createSubtreeIterator(ObjectReader reader)throws IncorrectObjectTypeException, IOException {if (currentSubtree == null)throw new IncorrectObjectTypeException(getEntryObjectId(),Constants.TYPE_TREE);return new DirCacheBuildIterator(this, currentSubtree);}
public GreekLowerCaseFilterFactory(Map<String,String> args) {super(args);if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public URI relativize(URI relative) {if (relative.opaque || opaque) {return relative;}if (scheme == null ? relative.scheme != null : !scheme.equals(relative.scheme)) {return relative;}if (authority == null ? relative.authority != null : !authority.equals(relative.authority)) {return relative;}String thisPath = normalize(path, false);String relativePath = normalize(relative.path, false);if (!thisPath.equals(relativePath)) {thisPath = thisPath.substring(0, thisPath.lastIndexOf('/') + 1);if (!relativePath.startsWith(thisPath)) {return relative;}}URI result = new URI();result.fragment = relative.fragment;result.query = relative.query;result.path = relativePath.substring(thisPath.length());result.setSchemeSpecificPart();return result;}
public Reader freeze(boolean trim) {if (frozen) {throw new IllegalStateException("already frozen");}if (didSkipBytes) {throw new IllegalStateException("cannot freeze when copy(BytesRef, BytesRef) was used");}if (trim && upto < blockSize) {final byte[] newBlock = new byte[upto];System.arraycopy(currentBlock, 0, newBlock, 0, upto);currentBlock = newBlock;}if (currentBlock == null) {currentBlock = EMPTY_BYTES;}addBlock(currentBlock);frozen = true;currentBlock = null;return new PagedBytes.Reader(this);}
public ValueEval evaluate(ValueEval[] args, OperationEvaluationContext ec) {if (args.length == 2) {return evaluate(ec.getRowIndex(), ec.getColumnIndex(), args[0], args[1]);}if (args.length == 3) {return evaluate(ec.getRowIndex(), ec.getColumnIndex(), args[0], args[1], args[2]);}return ErrorEval.VALUE_INVALID;}
public Cluster createCluster(CreateClusterRequest request) {request = beforeClientExecution(request);return executeCreateCluster(request);}
public PersistentSnapshotDeletionPolicy(IndexDeletionPolicy primary,Directory dir, OpenMode mode) throws IOException {super(primary);this.dir = dir;if (mode == OpenMode.CREATE) {clearPriorSnapshots();}loadPriorSnapshots();if (mode == OpenMode.APPEND && nextWriteGen == 0) {throw new IllegalStateException("no snapshots stored in this directory");}}
public String getText(RuleContext ctx) {return getText(ctx.getSourceInterval());}
public final float get() {if (position == limit) {throw new BufferUnderflowException();}return backingArray[offset + position++];}
public DeleteDataSetResult deleteDataSet(DeleteDataSetRequest request) {request = beforeClientExecution(request);return executeDeleteDataSet(request);}
public boolean contains(Object o) {return containsKey(o);}
public boolean matches(char s[], int len) {return super.matches(s, len) && !exceptions.contains(s, 0, len);}
public int getDeltaSearchWindowSize() {return deltaSearchWindowSize;}
public GetDomainNameResult getDomainName(GetDomainNameRequest request) {request = beforeClientExecution(request);return executeGetDomainName(request);}
public DeleteAccessLogSettingsResult deleteAccessLogSettings(DeleteAccessLogSettingsRequest request) {request = beforeClientExecution(request);return executeDeleteAccessLogSettings(request);}
public QueryValueSource(Query q, float defVal) {this.q = q;this.defVal = defVal;}
@Override public Object[] toArray() {return snapshot().toArray();}
public String toLexerString() {if ( s0==null ) return "";DFASerializer serializer = new LexerDFASerializer(this);return serializer.toString();}
public void clear() {fill(0, size(), 0);}
public GetStreamingDistributionConfigResult getStreamingDistributionConfig(GetStreamingDistributionConfigRequest request) {request = beforeClientExecution(request);return executeGetStreamingDistributionConfig(request);}
public UpdateDomainContactResult updateDomainContact(UpdateDomainContactRequest request) {request = beforeClientExecution(request);return executeUpdateDomainContact(request);}
public ListIterator<E> listIterator(int location) {return new LinkIterator<E>(this, location);}
public String toString() {StringBuilder buffer = new StringBuilder();buffer.append("[STARTBLOCK]\n");buffer.append("    .rt              =").append(HexDump.shortToHex(rt)).append('\n');buffer.append("    .grbitFrt        =").append(HexDump.shortToHex(grbitFrt)).append('\n');buffer.append("    .iObjectKind     =").append(HexDump.shortToHex(iObjectKind)).append('\n');buffer.append("    .iObjectContext  =").append(HexDump.shortToHex(iObjectContext)).append('\n');buffer.append("    .iObjectInstance1=").append(HexDump.shortToHex(iObjectInstance1)).append('\n');buffer.append("    .iObjectInstance2=").append(HexDump.shortToHex(iObjectInstance2)).append('\n');buffer.append("[/STARTBLOCK]\n");return buffer.toString();}
public long get(int index) {final int o = index / 7;final int b = index % 7;final int shift = b * 9;return (blocks[o] >>> shift) & 511L;}
public String toString(String field) {StringBuilder buffer = new StringBuilder();boolean needParens = (getLowFreqMinimumNumberShouldMatch() > 0);if (needParens) {buffer.append("(");}for (int i = 0; i < terms.size(); i++) {Term t = terms.get(i);buffer.append(newTermQuery(t, null).toString());if (i != terms.size() - 1) buffer.append(", ");}if (needParens) {buffer.append(")");}if (getLowFreqMinimumNumberShouldMatch() > 0 || getHighFreqMinimumNumberShouldMatch() > 0) {buffer.append('~');buffer.append("(");buffer.append(getLowFreqMinimumNumberShouldMatch());buffer.append(getHighFreqMinimumNumberShouldMatch());buffer.append(")");}return buffer.toString();}
public String[] getStopWords(String fieldName) {Set<String> stopWords = stopWordsPerField.get(fieldName);return stopWords != null ? stopWords.toArray(new String[stopWords.size()]) : new String[0];}
public void print(float f) {print(String.valueOf(f));}
public MopenCreateGroupRequest() {super("MoPen", "2018-02-11", "MopenCreateGroup", "mopen");setProtocol(ProtocolType.HTTPS);setMethod(MethodType.POST);}
public SmallObject(int type, byte[] data) {this.type = type;this.data = data;}
public final boolean matches(char c) {return Character.isUpperCase(c);}
public StartNotebookInstanceResult startNotebookInstance(StartNotebookInstanceRequest request) {request = beforeClientExecution(request);return executeStartNotebookInstance(request);}
public static void putUnicodeLE(String input, byte[] output, int offset) {byte[] bytes = input.getBytes(UTF16LE);System.arraycopy(bytes, 0, output, offset, bytes.length);}
public void deleteDocument(int docID) {final int i = readerIndex(docID);getSequentialSubReaders().get(i).deleteDocument(docID - readerBase(i));}
public boolean isRelevant(String docName, QualityQuery query) {QRelJudgement qrj = judgements.get(query.getQueryID());return qrj!=null && qrj.isRelevant(docName);}
public final int getBeginB() {return beginB;}
public ModifySpotFleetRequestResult modifySpotFleetRequest(ModifySpotFleetRequestRequest request) {request = beforeClientExecution(request);return executeModifySpotFleetRequest(request);}
public UncalcedRecord() {_reserved = 0;}
public static PageOrder valueOf(int value){return _table[value];}
public static CellValue valueOf(boolean booleanValue) {return booleanValue ? TRUE : FALSE;}
public void write(String str) {buf.append(str);}
public void addListener(HSSFListener lsnr, short sid) {List<HSSFListener> list = _records.computeIfAbsent(Short.valueOf(sid), k -> new ArrayList<>(1));list.add(lsnr);}
public GetMeetingResult getMeeting(GetMeetingRequest request) {request = beforeClientExecution(request);return executeGetMeeting(request);}
public void stopTimer() {stop = true;}
public AttachLoadBalancerTargetGroupsResult attachLoadBalancerTargetGroups(AttachLoadBalancerTargetGroupsRequest request) {request = beforeClientExecution(request);return executeAttachLoadBalancerTargetGroups(request);}
public GetQueryLoggingConfigResult getQueryLoggingConfig(GetQueryLoggingConfigRequest request) {request = beforeClientExecution(request);return executeGetQueryLoggingConfig(request);}
public ListIterator<E> listIterator() {Object[] snapshot = elements;return new CowIterator<E>(snapshot, 0, snapshot.length);}
public CreateSnapshotResult createSnapshot(CreateSnapshotRequest request) {request = beforeClientExecution(request);return executeCreateSnapshot(request);}
public boolean hasObject(AnyObjectId objectId) {try {return getObjectDatabase().has(objectId);} catch (IOException e) {throw new UncheckedIOException(e);}}
public final void sort(int from, int to) {checkRange(from, to);mergeSort(from, to);}
public <T extends EscherRecord> T getChildById( short recordId ) {for ( EscherRecord childRecord : this ) {if ( childRecord.getRecordId() == recordId ) {@SuppressWarnings( "unchecked" )final T result = (T) childRecord;return result;}}return null;}
public void serialize(LittleEndianOutput out) {out.writeShort(field_1_xBasis);out.writeShort(field_2_yBasis);out.writeShort(field_3_heightBasis);out.writeShort(field_4_scale);out.writeShort(field_5_indexToFontTable);}
public static String toHex(int value) {StringBuilder sb = new StringBuilder(8);writeHex(sb, value & 0xFFFFFFFFL, 8, "");return sb.toString();}
public static Collection<Ref> sort(Collection<Ref> refs) {final List<Ref> r = new ArrayList<>(refs);Collections.sort(r, INSTANCE);return r;}
public DescribeVpcsResult describeVpcs() {return describeVpcs(new DescribeVpcsRequest());}
public ListProposalsResult listProposals(ListProposalsRequest request) {request = beforeClientExecution(request);return executeListProposals(request);}
public void close() throws IOException {flush();output.close();}
public final T get() {return object;}
public BundleInstanceRequest(String instanceId, Storage storage) {setInstanceId(instanceId);setStorage(storage);}
public void back(int delta) {if (delta == 1 && 0 <= prevPtr) {currPtr = prevPtr;prevPtr = -1;if (!eof())parseEntry();return;} else if (delta <= 0)throw new ArrayIndexOutOfBoundsException(delta);final int[] trace = new int[delta + 1];Arrays.fill(trace, -1);int ptr = 0;while (ptr != currPtr) {System.arraycopy(trace, 1, trace, 0, delta);trace[delta] = ptr;while (raw[ptr] != 0)ptr++;ptr += OBJECT_ID_LENGTH + 1;}if (trace[1] == -1)throw new ArrayIndexOutOfBoundsException(delta);prevPtr = trace[0];currPtr = trace[1];parseEntry();}
public String toString() {return "pred_"+ruleIndex+":"+predIndex;}
public PatternSyntaxException(String description, String pattern, int index) {this.desc = description;this.pattern = pattern;this.index = index;}
public AlphaAnimation(float from, float to) {mStartAlpha = from;mEndAlpha = to;mCurrentAlpha = from;}
public int doLogic() throws Exception {TaxonomyWriter taxonomyWriter = getRunData().getTaxonomyWriter();if (taxonomyWriter != null) {taxonomyWriter.commit();} else {throw new IllegalStateException("TaxonomyWriter is not currently open");}return 1;}
public DeltaIndex(byte[] sourceBuffer) {src = sourceBuffer;DeltaIndexScanner scan = new DeltaIndexScanner(src, src.length);table = scan.table;tableMask = scan.tableMask;entries = new long[1 + countEntries(scan)];copyEntries(scan);}
public int previousIndex() {return pos;}
public QueryMaker getQueryMaker() {return getRunData().getQueryMaker(this);}
public JapaneseTokenizerFactory(Map<String,String> args) {super(args);mode = Mode.valueOf(get(args, MODE, JapaneseTokenizer.DEFAULT_MODE.toString()).toUpperCase(Locale.ROOT));userDictionaryPath = args.remove(USER_DICT_PATH);userDictionaryEncoding = args.remove(USER_DICT_ENCODING);discardPunctuation = getBoolean(args, DISCARD_PUNCTUATION, true);discardCompoundToken = getBoolean(args, DISCARD_COMPOUND_TOKEN, true);nbestCost = getInt(args, NBEST_COST, 0);nbestExamples = args.remove(NBEST_EXAMPLES);if (!args.isEmpty()) {throw new IllegalArgumentException("Unknown parameters: " + args);}}
public Long longValue(String key) {String value = responseMap.get(key);if (null == value || 0 == value.length()) {return null;}return Long.valueOf(responseMap.get(key));}
public GetLibraryRequest() {super("CloudPhoto", "2017-07-11", "GetLibrary", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public short getFontOfFormattingRun(int index) {FormatRun r = _string.getFormatRun(index);return r.getFontIndex();}
public int fillFields(byte[] data, int offset, EscherRecordFactory recordFactory) {int bytesRemaining = readHeader( data, offset );int pos            = offset + 8;int size           = 0;field_1_shapeIdMax     =  LittleEndian.getInt( data, pos + size );size+=4;size+=4;field_3_numShapesSaved =  LittleEndian.getInt( data, pos + size );size+=4;field_4_drawingsSaved  =  LittleEndian.getInt( data, pos + size );size+=4;field_5_fileIdClusters.clear();int numIdClusters = (bytesRemaining-size) / 8;for (int i = 0; i < numIdClusters; i++) {int drawingGroupId = LittleEndian.getInt( data, pos + size );int numShapeIdsUsed = LittleEndian.getInt( data, pos + size + 4 );FileIdCluster fic = new FileIdCluster(drawingGroupId, numShapeIdsUsed);field_5_fileIdClusters.add(fic);maxDgId = Math.max(maxDgId, drawingGroupId);size += 8;}bytesRemaining -= size;if (bytesRemaining != 0) {throw new RecordFormatException("Expecting no remaining data but got " + bytesRemaining + " byte(s).");}return 8 + size;}
public void encode(int[] values, int valuesOffset, byte[] blocks,int blocksOffset, int iterations) {for (int i = 0; i < iterations; ++i) {final long block = encode(values, valuesOffset);valuesOffset += valueCount;blocksOffset = writeLong(block, blocks, blocksOffset);}}
public GetTerminologyResult getTerminology(GetTerminologyRequest request) {request = beforeClientExecution(request);return executeGetTerminology(request);}
public void serialize(LittleEndianOutput out) {out.writeShort(_character);out.writeShort(_fontIndex);}
public void serialize(LittleEndianOutput out) {out.writeShort(field_1_options);}
public SearchFacesResult searchFaces(SearchFacesRequest request) {request = beforeClientExecution(request);return executeSearchFaces(request);}
public int getPositionIncrementGap(String fieldName) {return getWrappedAnalyzer(fieldName).getPositionIncrementGap(fieldName);}
public DescribeSchemaResult describeSchema(DescribeSchemaRequest request) {request = beforeClientExecution(request);return executeDescribeSchema(request);}
@Override public int size() {return BoundedMap.this.size();}
public MutableEntry cloneEntry() {final MutableEntry r = new MutableEntry();ensureId();r.idBuffer.fromObjectId(idBuffer);r.offset = offset;return r;}
public OperateEquipmentRequest() {super("industry-brain", "2018-07-12", "OperateEquipment");setProtocol(ProtocolType.HTTPS);setMethod(MethodType.POST);}
public boolean add(E e) {synchronized (mutex) {return delegate().add(e);}}
public boolean equals( Object o ){if ( this == o ) {return true;}if ( !( o instanceof EscherSimpleProperty ) ) {return false;}final EscherSimpleProperty escherSimpleProperty = (EscherSimpleProperty) o;if ( propertyValue != escherSimpleProperty.propertyValue ) {return false;}if ( getId() != escherSimpleProperty.getId() ) {return false;}return true;}
public final FloatBuffer asFloatBuffer() {return FloatToByteBufferAdapter.asFloatBuffer(this);}
public void removeThumbnail() {remove1stProperty(PropertyIDMap.PID_THUMBNAIL);}
public static int compareIgnoreCase(String a, String b) {for (int i = 0; i < a.length() && i < b.length(); i++) {int d = toLowerCase(a.charAt(i)) - toLowerCase(b.charAt(i));if (d != 0)return d;}return a.length() - b.length();}
public ViewDefinitionRecord(RecordInputStream in) {rwFirst = in.readUShort();rwLast = in.readUShort();colFirst = in.readUShort();colLast = in.readUShort();rwFirstHead = in.readUShort();rwFirstData = in.readUShort();colFirstData = in.readUShort();iCache = in.readUShort();reserved = in.readUShort();sxaxis4Data = in.readUShort();ipos4Data = in.readUShort();cDim = in.readUShort();cDimRw = in.readUShort();cDimCol = in.readUShort();cDimPg = in.readUShort();cDimData = in.readUShort();cRw = in.readUShort();cCol = in.readUShort();grbit = in.readUShort();itblAutoFmt = in.readUShort();int cchName = in.readUShort();int cchData = in.readUShort();name = StringUtil.readUnicodeString(in, cchName);dataField = StringUtil.readUnicodeString(in, cchData);}
public FormatRecord(RecordInputStream in) {field_1_index_code = in.readShort();int field_3_unicode_len = in.readUShort();field_3_hasMultibyte = (in.readByte() & 0x01) != 0;if (field_3_hasMultibyte) {field_4_formatstring = readStringCommon(in, field_3_unicode_len, false);} else {field_4_formatstring = readStringCommon(in, field_3_unicode_len, true);}}
public DescribeBrokerResult describeBroker(DescribeBrokerRequest request) {request = beforeClientExecution(request);return executeDescribeBroker(request);}
public void reset() {if ( getInputStream()!=null ) getInputStream().seek(0);_errHandler.reset(this);_ctx = null;_syntaxErrors = 0;matchedEOF = false;setTrace(false);_precedenceStack.clear();_precedenceStack.push(0);ATNSimulator interpreter = getInterpreter();if (interpreter != null) {interpreter.reset();}}
public boolean remove(Object o) {final RevFlag flag = (RevFlag) o;if ((mask & flag.mask) == 0)return false;mask &= ~flag.mask;for (int i = 0; i < active.size(); i++)if (active.get(i).mask == flag.mask)active.remove(i);return true;}
public String format(Passage passages[], String content) {StringBuilder sb = new StringBuilder();int pos = 0;for (Passage passage : passages) {if (passage.getStartOffset() > pos && pos > 0) {sb.append(ellipsis);}pos = passage.getStartOffset();for (int i = 0; i < passage.getNumMatches(); i++) {int start = passage.getMatchStarts()[i];assert start >= pos && start < passage.getEndOffset();append(sb, content, pos, start);int end = passage.getMatchEnds()[i];assert end > start;while (i + 1 < passage.getNumMatches() && passage.getMatchStarts()[i+1] < end) {end = passage.getMatchEnds()[++i];}end = Math.min(end, passage.getEndOffset()); sb.append(preTag);append(sb, content, start, end);sb.append(postTag);pos = end;}append(sb, content, pos, Math.max(pos, passage.getEndOffset()));pos = passage.getEndOffset();}return sb.toString();}
public DrillSidewaysResult(Facets facets, TopDocs hits) {this.facets = facets;this.hits = hits;}
public ListTrafficPolicyInstancesByPolicyResult listTrafficPolicyInstancesByPolicy(ListTrafficPolicyInstancesByPolicyRequest request) {request = beforeClientExecution(request);return executeListTrafficPolicyInstancesByPolicy(request);}
public ComplexPhraseQuery(String field, String phrasedQueryStringContents,int slopFactor, boolean inOrder) {this.field = Objects.requireNonNull(field);this.phrasedQueryStringContents = Objects.requireNonNull(phrasedQueryStringContents);this.slopFactor = slopFactor;this.inOrder = inOrder;}
public String toString(String field) {StringBuilder buffer = new StringBuilder();if (!term.field().equals(field)) {buffer.append(term.field());buffer.append(":");}buffer.append(getClass().getSimpleName());buffer.append(" {");buffer.append('\n');buffer.append(automaton.toString());buffer.append("}");return buffer.toString();}
public final String toFormulaString() {return getName();}
public AreaRecord clone() {return copy();}
public long ramBytesUsed() {return TERMS_BASE_RAM_BYTES_USED + (fst!=null ? fst.ramBytesUsed() : 0)+ RamUsageEstimator.sizeOf(scratch.bytes()) + RamUsageEstimator.sizeOf(scratchUTF16.chars());}
public DeleteConfigurationTemplateRequest(String applicationName, String templateName) {setApplicationName(applicationName);setTemplateName(templateName);}
public List<Token> getTokens(int start, int stop, int ttype) {HashSet<Integer> s = new HashSet<Integer>(ttype);s.add(ttype);return getTokens(start,stop, s);}
public DescribeIamInstanceProfileAssociationsResult describeIamInstanceProfileAssociations(DescribeIamInstanceProfileAssociationsRequest request) {request = beforeClientExecution(request);return executeDescribeIamInstanceProfileAssociations(request);}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval textArg) {ValueEval veText1;try {veText1 = OperandResolver.getSingleValue(textArg, srcRowIndex, srcColumnIndex);} catch (EvaluationException e) {return e.getErrorEval();}String text = OperandResolver.coerceValueToString(veText1);if (text.length() == 0) {return ErrorEval.VALUE_INVALID;}int code = text.charAt(0);return new StringEval(String.valueOf(code));}
public AttachVpnGatewayResult attachVpnGateway(AttachVpnGatewayRequest request) {request = beforeClientExecution(request);return executeAttachVpnGateway(request);}
public int compareTo(FloatBuffer otherBuffer) {int compareRemaining = (remaining() < otherBuffer.remaining()) ? remaining(): otherBuffer.remaining();int thisPos = position;int otherPos = otherBuffer.position;float thisFloat, otherFloat;while (compareRemaining > 0) {thisFloat = get(thisPos);otherFloat = otherBuffer.get(otherPos);if ((thisFloat != otherFloat)&& ((thisFloat == thisFloat) || (otherFloat == otherFloat))) {return thisFloat < otherFloat ? -1 : 1;}thisPos++;otherPos++;compareRemaining--;}return remaining() - otherBuffer.remaining();}
public Matcher useTransparentBounds(boolean value) {transparentBounds = value;useTransparentBoundsImpl(address, value);return this;}
public void remove() {if (lastEntryReturned == null)throw new IllegalStateException();if (modCount != expectedModCount)throw new ConcurrentModificationException();Hashtable.this.remove(lastEntryReturned.key);lastEntryReturned = null;expectedModCount = modCount;}
public String toFormulaString() {StringBuilder sb = new StringBuilder(64);if (externalWorkbookNumber >= 0) {sb.append('[');sb.append(externalWorkbookNumber);sb.append(']');}if (sheetName != null) {SheetNameFormatter.appendFormat(sb, sheetName);}sb.append('!');sb.append(FormulaError.REF.getString());return sb.toString();}
public String toString() {return slice.toString()+":"+ postingsEnum;}
public CreateVpnConnectionRouteResult createVpnConnectionRoute(CreateVpnConnectionRouteRequest request) {request = beforeClientExecution(request);return executeCreateVpnConnectionRoute(request);}
public boolean hasNext() {return next != null;}
public DeleteDBSecurityGroupRequest(String dBSecurityGroupName) {setDBSecurityGroupName(dBSecurityGroupName);}
public int compare(Property o1, Property o2){String VBA_PROJECT = "_VBA_PROJECT";String name1  = o1.getName();String name2  = o2.getName();int  result = name1.length() - name2.length();if (result == 0){if (name1.compareTo(VBA_PROJECT) == 0)result = 1;else if (name2.compareTo(VBA_PROJECT) == 0)result = -1;else{if (name1.startsWith("__") && name2.startsWith("__")){result = name1.compareToIgnoreCase(name2);}else if (name1.startsWith("__")){result = 1;}else if (name2.startsWith("__")){result = -1;}elseresult = name1.compareToIgnoreCase(name2);}}return result;}
public DoubleBuffer get(double[] dst, int dstOffset, int doubleCount) {Arrays.checkOffsetAndCount(dst.length, dstOffset, doubleCount);if (doubleCount > remaining()) {throw new BufferUnderflowException();}for (int i = dstOffset; i < dstOffset + doubleCount; ++i) {dst[i] = get();}return this;}
public CharsRef add(CharsRef prefix, CharsRef output) {assert prefix != null;assert output != null;if (prefix == NO_OUTPUT) {return output;} else if (output == NO_OUTPUT) {return prefix;} else {assert prefix.length > 0;assert output.length > 0;CharsRef result = new CharsRef(prefix.length + output.length);System.arraycopy(prefix.chars, prefix.offset, result.chars, 0, prefix.length);System.arraycopy(output.chars, output.offset, result.chars, prefix.length, output.length);result.length = prefix.length + output.length;return result;}}
public UpdateProfileResult updateProfile(UpdateProfileRequest request) {request = beforeClientExecution(request);return executeUpdateProfile(request);}
public LikeThisQueryBuilder(Analyzer analyzer, String[] defaultFieldNames) {this.analyzer = analyzer;this.defaultFieldNames = defaultFieldNames;}
public StringBuffer insert(int index, long l) {return insert(index, Long.toString(l));}
public Field(String name, BytesRef bytes, IndexableFieldType type) {if (name == null) {throw new IllegalArgumentException("name must not be null");}if (bytes == null) {throw new IllegalArgumentException("bytes must not be null");}if (type == null) {throw new IllegalArgumentException("type must not be null");}this.name = name;this.fieldsData = bytes;this.type = type;}
public void clear() {mSize = 0;}
public SrndQuery parse2(String query) throws ParseException {ReInit(new FastCharStream(new StringReader(query)));try {return TopSrndQuery();} catch (TokenMgrError tme) {throw new ParseException(tme.getMessage());}}
@Override public int size() {return (int) Math.min(this.size, Integer.MAX_VALUE);}
public DescribeConfigurationResult describeConfiguration(DescribeConfigurationRequest request) {request = beforeClientExecution(request);return executeDescribeConfiguration(request);}
public String getCharErrorDisplay(int c) {String s = getErrorDisplay(c);return "'"+s+"'";}
public DescribeHumanTaskUiResult describeHumanTaskUi(DescribeHumanTaskUiRequest request) {request = beforeClientExecution(request);return executeDescribeHumanTaskUi(request);}
public void run() {try {int n = task.runAndMaybeStats(letChildReport);if (anyExhaustibleTasks) {updateExhausted(task);}count += n;} catch (NoMoreDataException e) {exhausted = true;} catch (Exception e) {throw new RuntimeException(e);}}
public DescribeImagePermissionsResult describeImagePermissions(DescribeImagePermissionsRequest request) {request = beforeClientExecution(request);return executeDescribeImagePermissions(request);}
public SrndQuery clone() {try {return (SrndQuery)super.clone();} catch (CloneNotSupportedException cns) {throw new Error(cns);}}
public void recycleByteBlocks(byte[][] blocks, int start, int end) {final int numBlocks = Math.min(maxBufferedBlocks - freeBlocks, end - start);final int size = freeBlocks + numBlocks;if (size >= freeByteBlocks.length) {final byte[][] newBlocks = new byte[ArrayUtil.oversize(size,RamUsageEstimator.NUM_BYTES_OBJECT_REF)][];System.arraycopy(freeByteBlocks, 0, newBlocks, 0, freeBlocks);freeByteBlocks = newBlocks;}final int stop = start + numBlocks;for (int i = start; i < stop; i++) {freeByteBlocks[freeBlocks++] = blocks[i];blocks[i] = null;}for (int i = stop; i < end; i++) {blocks[i] = null;}bytesUsed.addAndGet(-(end - stop) * blockSize);assert bytesUsed.get() >= 0;}
public GeohashPrefixTree(SpatialContext ctx, int maxLevels) {super(ctx, maxLevels);Rectangle bounds = ctx.getWorldBounds();if (bounds.getMinX() != -180)throw new IllegalArgumentException("Geohash only supports lat-lon world bounds. Got "+bounds);int MAXP = getMaxLevelsPossible();if (maxLevels <= 0 || maxLevels > MAXP)throw new IllegalArgumentException("maxLevels must be [1-"+MAXP+"] but got "+ maxLevels);}
public void removeName(int namenum) {_definedNames.remove(namenum);}
public CancelSpotFleetRequestsResult cancelSpotFleetRequests(CancelSpotFleetRequestsRequest request) {request = beforeClientExecution(request);return executeCancelSpotFleetRequests(request);}
public GetIndustryInfoLineageListRequest() {super("industry-brain", "2018-07-12", "GetIndustryInfoLineageList");setProtocol(ProtocolType.HTTPS);}
public static double[] grow(double[] array, int minSize) {assert minSize >= 0: "size must be positive (got " + minSize + "): likely integer overflow?";if (array.length < minSize) {return growExact(array, oversize(minSize, Double.BYTES));} elsereturn array;}
public void setResult(RefUpdate.Result status) {result = status;super.setResult(status);}
public void decode(byte[] blocks, int blocksOffset, int[] values, int valuesOffset, int iterations) {for (int i = 0; i < iterations; ++i) {final int byte0 = blocks[blocksOffset++] & 0xFF;final int byte1 = blocks[blocksOffset++] & 0xFF;values[valuesOffset++] = (byte0 << 2) | (byte1 >>> 6);final int byte2 = blocks[blocksOffset++] & 0xFF;values[valuesOffset++] = ((byte1 & 63) << 4) | (byte2 >>> 4);final int byte3 = blocks[blocksOffset++] & 0xFF;values[valuesOffset++] = ((byte2 & 15) << 6) | (byte3 >>> 2);final int byte4 = blocks[blocksOffset++] & 0xFF;values[valuesOffset++] = ((byte3 & 3) << 8) | byte4;}}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval arg0, ValueEval arg1, ValueEval arg2) {try {ValueEval ve = OperandResolver.getSingleValue(arg0, srcRowIndex, srcColumnIndex);final double result = OperandResolver.coerceValueToDouble(ve);if (Double.isNaN(result) || Double.isInfinite(result)) {throw new EvaluationException(ErrorEval.NUM_ERROR);}ve = OperandResolver.getSingleValue(arg2, srcRowIndex, srcColumnIndex);int order_value = OperandResolver.coerceValueToInt(ve);final boolean order;if (order_value==0) {order = true;} else if(order_value==1) {order = false;} else {throw new EvaluationException(ErrorEval.NUM_ERROR);}if (arg1 instanceof RefListEval) {return eval(result, ((RefListEval)arg1), order);}final AreaEval aeRange = convertRangeArg(arg1);return eval(result, aeRange, order);} catch (EvaluationException e) {return e.getErrorEval();}}
public DeleteEventBusResult deleteEventBus(DeleteEventBusRequest request) {request = beforeClientExecution(request);return executeDeleteEventBus(request);}
public static ByteBuffer wrap(byte[] array, int start, int byteCount) {Arrays.checkOffsetAndCount(array.length, start, byteCount);ByteBuffer buf = new ReadWriteHeapByteBuffer(array);buf.position = start;buf.limit = start + byteCount;return buf;}
public String apiVersion() {return this.apiVersion;}
public SearchResult search(SearchRequest request) {request = beforeClientExecution(request);return executeSearch(request);}
public PushCommand setRemote(String remote) {checkCallable();this.remote = remote;return this;}
public AcceptReservedInstancesExchangeQuoteResult acceptReservedInstancesExchangeQuote(AcceptReservedInstancesExchangeQuoteRequest request) {request = beforeClientExecution(request);return executeAcceptReservedInstancesExchangeQuote(request);}
public GetAuthorizationTokenResult getAuthorizationToken(GetAuthorizationTokenRequest request) {request = beforeClientExecution(request);return executeGetAuthorizationToken(request);}
public static InitCommand init() {return new InitCommand();}
public static RevFilter create(Collection<RevFilter> list) {if (list.size() < 2)throw new IllegalArgumentException(JGitText.get().atLeastTwoFiltersNeeded);final RevFilter[] subfilters = new RevFilter[list.size()];list.toArray(subfilters);if (subfilters.length == 2)return create(subfilters[0], subfilters[1]);return new List(subfilters);}
public static PredictionContext mergeRoot(SingletonPredictionContext a,SingletonPredictionContext b,boolean rootIsWildcard){if ( rootIsWildcard ) {if ( a == EMPTY ) return EMPTY;  if ( b == EMPTY ) return EMPTY;  }else {if ( a == EMPTY && b == EMPTY ) return EMPTY; if ( a == EMPTY ) { int[] payloads = {b.returnState, EMPTY_RETURN_STATE};PredictionContext[] parents = {b.parent, null};PredictionContext joined =new ArrayPredictionContext(parents, payloads);return joined;}if ( b == EMPTY ) { int[] payloads = {a.returnState, EMPTY_RETURN_STATE};PredictionContext[] parents = {a.parent, null};PredictionContext joined =new ArrayPredictionContext(parents, payloads);return joined;}}return null;}
public ListTerminologiesResult listTerminologies(ListTerminologiesRequest request) {request = beforeClientExecution(request);return executeListTerminologies(request);}
public ModifyInstanceGroupsRequest(java.util.List<InstanceGroupModifyConfig> instanceGroups) {setInstanceGroups(instanceGroups);}
public String toString() {return "AnyObjectId[" + name() + "]";}
public long ramBytesUsed() {long ramBytesUsed = postingsReader.ramBytesUsed();for (TermsReader r : fields.values()) {ramBytesUsed += r.ramBytesUsed();}return ramBytesUsed;}
public static final ObjectId fromRaw(int[] is, int p) {return new ObjectId(is[p], is[p + 1], is[p + 2], is[p + 3], is[p + 4]);}
public RemoveTagsFromStreamResult removeTagsFromStream(RemoveTagsFromStreamRequest request) {request = beforeClientExecution(request);return executeRemoveTagsFromStream(request);}
public void writeChar(int value) throws IOException {checkWritePrimitiveTypes();primitiveTypes.writeChar(value);}
public void setParams(String params) {super.setParams(params);if (params != null) {commitUserData = params;}}
public OptionGroup modifyOptionGroup(ModifyOptionGroupRequest request) {request = beforeClientExecution(request);return executeModifyOptionGroup(request);}
public CreateCommentResult createComment(CreateCommentRequest request) {request = beforeClientExecution(request);return executeCreateComment(request);}
public void setParams(String params) {super.setParams(params);userData = params;}
public SearchAvailablePhoneNumbersResult searchAvailablePhoneNumbers(SearchAvailablePhoneNumbersRequest request) {request = beforeClientExecution(request);return executeSearchAvailablePhoneNumbers(request);}
public SpanPositionCheckQuery(SpanQuery match) {this.match = Objects.requireNonNull(match);}
public boolean removeChildRecord(EscherRecord toBeRemoved) {return _childRecords.remove(toBeRemoved);}
public BytesRef clone() {return new BytesRef(bytes, offset, length);}
public ByteBuffer putLong(long value) {throw new ReadOnlyBufferException();}
@Override public boolean add(E object) {synchronized (CopyOnWriteArrayList.this) {add(slice.to - slice.from, object);return true;}}
public RevTree lookupTree(AnyObjectId id) {RevTree c = (RevTree) objects.get(id);if (c == null) {c = new RevTree(id);objects.add(c);}return c;}
public boolean equals(Object other) {return sameClassAs(other) &&func.equals(((FunctionQuery) other).func);}
public boolean changeExternalReference(String oldUrl, String newUrl) {for (ExternalBookBlock ex : _externalBookBlocks) {SupBookRecord externalRecord = ex.getExternalBookRecord();if (externalRecord.isExternalReferences()&& externalRecord.getURL().equals(oldUrl)) {externalRecord.setURL(newUrl);return true;}}return false;}
public void removeLastPrinted() {remove1stProperty(PropertyIDMap.PID_LASTPRINTED);}
public MergeCommand merge() {return new MergeCommand(repo);}
public String toString() {final Type t = getType();return t + "(" + beginA + "-" + endA + "," + beginB + "-" + endB + ")";}
public void serialize(LittleEndianOutput out) {int nItems = _list.size();out.writeShort(nItems);for (int k = 0; k < nItems; k++) {CellRangeAddress region = _list.get(k);region.serialize(out);}}
public void remove() {throw new UnsupportedOperationException("Remove not supported");}
public TagCommand setSigned(boolean signed) {this.signed = signed;return this;}
public DescribeReservedInstancesListingsResult describeReservedInstancesListings(DescribeReservedInstancesListingsRequest request) {request = beforeClientExecution(request);return executeDescribeReservedInstancesListings(request);}
public String getName() {return getRef().getName();}
public boolean isAllSet(final int holder){return (holder & _mask) == _mask;}
public static int getEncodedSize(String value) {int result = 2 + 1;result += value.length() * (StringUtil.hasMultibyte(value) ? 2 : 1);return result;}
public List<CharsRef> stem(char word[], int length) {if (dictionary.needsInputCleaning) {scratchSegment.setLength(0);scratchSegment.append(word, 0, length);CharSequence cleaned = dictionary.cleanInput(scratchSegment, segment);scratchBuffer = ArrayUtil.grow(scratchBuffer, cleaned.length());length = segment.length();segment.getChars(0, length, scratchBuffer, 0);word = scratchBuffer;}int caseType = caseOf(word, length);if (caseType == UPPER_CASE) {caseFoldTitle(word, length);caseFoldLower(titleBuffer, length);List<CharsRef> list = doStem(word, length, false);list.addAll(doStem(titleBuffer, length, true));list.addAll(doStem(lowerBuffer, length, true));return list;} else if (caseType == TITLE_CASE) {caseFoldLower(word, length);List<CharsRef> list = doStem(word, length, false);list.addAll(doStem(lowerBuffer, length, true));return list;} else {return doStem(word, length, false);}}
public HSSFConditionalFormattingRule createConditionalFormattingRule(String formula) {CFRuleRecord rr = CFRuleRecord.create(_sheet, formula);return new HSSFConditionalFormattingRule(_sheet, rr);}
public Record create(RecordInputStream in) {Object[] args = { in, };try {return (org.apache.poi.hssf.record.Record) _m.invoke(null, args);} catch (IllegalArgumentException | IllegalAccessException e) {throw new RuntimeException(e);} catch (InvocationTargetException e) {throw new org.apache.poi.util.RecordFormatException("Unable to construct record instance" , e.getTargetException());}}
public int set(int index, long[] arr, int off, int len) {assert len > 0 : "len must be > 0 (got " + len + ")";assert index >= 0 && index < size();len = Math.min(len, size() - index);assert off + len <= arr.length;for (int i = index, o = off, end = index + len; i < end; ++i, ++o) {set(i, arr[o]);}return len;}
public synchronized long ramBytesUsed() {long bytes = 0;for(CachedOrds ords : ordsCache.values()) {bytes += ords.ramBytesUsed();}return bytes;}
public void writeDouble(double v) {writeLong(Double.doubleToLongBits(v));}
public String toString() {return "DocumentsWriterFlushControl [activeBytes=" + activeBytes+ ", flushBytes=" + flushBytes + "]";}
public ListSecurityConfigurationsResult listSecurityConfigurations(ListSecurityConfigurationsRequest request) {request = beforeClientExecution(request);return executeListSecurityConfigurations(request);}
public ListQualificationRequestsResult listQualificationRequests(ListQualificationRequestsRequest request) {request = beforeClientExecution(request);return executeListQualificationRequests(request);}
public void println(char[] chars) {println(new String(chars, 0, chars.length));}
public ReleaseAddressResult releaseAddress(ReleaseAddressRequest request) {request = beforeClientExecution(request);return executeReleaseAddress(request);}
public static boolean[] copyOfRange(boolean[] original, int start, int end) {if (start > end) {throw new IllegalArgumentException();}int originalLength = original.length;if (start < 0 || start > originalLength) {throw new ArrayIndexOutOfBoundsException();}int resultLength = end - start;int copyLength = Math.min(resultLength, originalLength - start);boolean[] result = new boolean[resultLength];System.arraycopy(original, start, result, 0, copyLength);return result;}
public void fillOval(int x, int y, int width, int height){HSSFSimpleShape shape = escherGroup.createShape(new HSSFChildAnchor( x, y, x + width, y + height ) );shape.setShapeType(HSSFSimpleShape.OBJECT_TYPE_OVAL);shape.setLineStyle(HSSFShape.LINESTYLE_NONE);shape.setFillColor(foreground.getRed(), foreground.getGreen(), foreground.getBlue());shape.setLineStyleColor(foreground.getRed(), foreground.getGreen(), foreground.getBlue());shape.setNoFill(false);}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval arg0, ValueEval arg1,ValueEval arg2) {try {String needle = TextFunction.evaluateStringArg(arg0, srcRowIndex, srcColumnIndex);String haystack = TextFunction.evaluateStringArg(arg1, srcRowIndex, srcColumnIndex);int startpos = TextFunction.evaluateIntArg(arg2, srcRowIndex, srcColumnIndex) - 1;if (startpos < 0) {return ErrorEval.VALUE_INVALID;}return eval(haystack, needle, startpos);} catch (EvaluationException e) {return e.getErrorEval();}}
public CreateInvalidationRequest(String distributionId, InvalidationBatch invalidationBatch) {setDistributionId(distributionId);setInvalidationBatch(invalidationBatch);}
public CreateUsageReportSubscriptionResult createUsageReportSubscription(CreateUsageReportSubscriptionRequest request) {request = beforeClientExecution(request);return executeCreateUsageReportSubscription(request);}
public static String fromString(String value) {return value;}
public GetDetectorsResult getDetectors(GetDetectorsRequest request) {request = beforeClientExecution(request);return executeGetDetectors(request);}
public static String fromDouble(Double d) {return Double.toString(d);}
public void writeProtectWorkbook( String password, String username ) {FileSharingRecord frec = getFileSharing();WriteAccessRecord waccess = getWriteAccess(); getWriteProtect();frec.setReadOnly((short)1);frec.setPassword((short)CryptoFunctions.createXorVerifier1(password));frec.setUsername(username);waccess.setUsername(username);}
public Process exec(String command, int timeout)throws TransportException {String ssh = SystemReader.getInstance().getenv("GIT_SSH"); boolean putty = ssh.toLowerCase(Locale.ROOT).contains("plink"); List<String> args = new ArrayList<>();args.add(ssh);if (putty&& !ssh.toLowerCase(Locale.ROOT).contains("tortoiseplink")) args.add("-batch"); if (0 < getURI().getPort()) {args.add(putty ? "-P" : "-p"); args.add(String.valueOf(getURI().getPort()));}if (getURI().getUser() != null)args.add(getURI().getUser() + "@" + getURI().getHost()); elseargs.add(getURI().getHost());args.add(command);ProcessBuilder pb = createProcess(args);try {return pb.start();} catch (IOException err) {throw new TransportException(err.getMessage(), err);}}
public void serialize(LittleEndianOutput out) {out.write(recordData);}
public UpdateFleetCapacityResult updateFleetCapacity(UpdateFleetCapacityRequest request) {request = beforeClientExecution(request);return executeUpdateFleetCapacity(request);}
public CreateDirectConnectGatewayAssociationResult createDirectConnectGatewayAssociation(CreateDirectConnectGatewayAssociationRequest request) {request = beforeClientExecution(request);return executeCreateDirectConnectGatewayAssociation(request);}
public TokenStream create(TokenStream input) {if (words == null) {return input;} else {final TokenStream filter = new KeepWordFilter(input, words);return filter;}}
public final int getEndA() {return endA;}
public String getStrictHostKeyChecking() {return strictHostKeyChecking;}
public Lift(boolean changeSkip) {this.changeSkip = changeSkip;}
public void serialize(LittleEndianOutput out) {out.writeShort(field_1_precision);}
public GetAuthorizerResult getAuthorizer(GetAuthorizerRequest request) {request = beforeClientExecution(request);return executeGetAuthorizer(request);}
public StringCharacterIterator(String value, int start, int end, int location) {string = value;if (start < 0 || end > string.length() || start > end|| location < start || location > end) {throw new IllegalArgumentException();}this.start = start;this.end = end;offset = location;}
public String toString() {StringBuilder buf = new StringBuilder();buf.append("ObjectToPack[");buf.append(Constants.typeString(getType()));buf.append(" ");buf.append(name());if (wantWrite())buf.append(" wantWrite");if (isReuseAsIs())buf.append(" reuseAsIs");if (isDoNotDelta())buf.append(" doNotDelta");if (isEdge())buf.append(" edge");if (getDeltaDepth() > 0)buf.append(" depth=").append(getDeltaDepth());if (isDeltaRepresentation()) {if (getDeltaBase() != null)buf.append(" base=inpack:").append(getDeltaBase().name());elsebuf.append(" base=edge:").append(getDeltaBaseId().name());}if (isWritten())buf.append(" offset=").append(getOffset());buf.append("]");return buf.toString();}
public String toString() {return "1";}
public final void readFully(byte[] dst, int offset, int byteCount) throws IOException {Streams.readFully(in, dst, offset, byteCount);}
public GetMailboxDetailsResult getMailboxDetails(GetMailboxDetailsRequest request) {request = beforeClientExecution(request);return executeGetMailboxDetails(request);}
public CharBuffer append(CharSequence csq) {if (csq != null) {return put(csq.toString());}return put("null");}
public RegisterFaceRequest() {super("LinkFace", "2018-07-20", "RegisterFace");setProtocol(ProtocolType.HTTPS);setMethod(MethodType.POST);}
public static void checkValue(double result) throws EvaluationException {if (Double.isNaN(result) || Double.isInfinite(result)) {throw new EvaluationException(ErrorEval.NUM_ERROR);}}
public PutInvitationConfigurationResult putInvitationConfiguration(PutInvitationConfigurationRequest request) {request = beforeClientExecution(request);return executePutInvitationConfiguration(request);}
public QueryNode process(QueryNode queryTree) throws QueryNodeException {Operator op = getQueryConfigHandler().get(ConfigurationKeys.DEFAULT_OPERATOR);if (op == null) {throw new IllegalArgumentException("StandardQueryConfigHandler.ConfigurationKeys.DEFAULT_OPERATOR should be set on the QueryConfigHandler");}this.usingAnd = StandardQueryConfigHandler.Operator.AND == op;return super.process(queryTree);}
public void add(BytesRef utf8, int bucket) throws IOException {if (bucket < 0 || bucket >= buckets) {throw new IllegalArgumentException("Bucket outside of the allowed range [0, " + buckets + "): " + bucket);}scratch.grow(utf8.length + 10);scratch.clear();scratch.append((byte) bucket);scratch.append(utf8);sorter.add(scratch.get());}
public DescribeWorkspaceBundlesResult describeWorkspaceBundles() {return describeWorkspaceBundles(new DescribeWorkspaceBundlesRequest());}
public static String decode(String s) {return decode(s, false, Charsets.UTF_8);}
public void setExpire(Date expire) {this.expire = expire;expireAgeMillis = -1;}
public int DecRef() {assert count > 0: Thread.currentThread().getName() + ": RefCount is 0 pre-decrement for file \"" + fileName + "\"";return --count;}
public List<WeightedFragInfo> getWeightedFragInfoList( List<WeightedFragInfo> src ) {return src;}
public CreateInstancesFromSnapshotResult createInstancesFromSnapshot(CreateInstancesFromSnapshotRequest request) {request = beforeClientExecution(request);return executeCreateInstancesFromSnapshot(request);}
public Comparator<? super E> comparator() {return backingMap.comparator();}
public boolean isValueSecure() {return valueSecure;}
public static short[] grow(short[] array, int minSize) {assert minSize >= 0: "size must be positive (got " + minSize + "): likely integer overflow?";if (array.length < minSize) {return growExact(array, oversize(minSize, Short.BYTES));} elsereturn array;}
public ObjectId idFor(int type, byte[] data) {return idFor(type, data, 0, data.length);}
public CreateDomainNameResult createDomainName(CreateDomainNameRequest request) {request = beforeClientExecution(request);return executeCreateDomainName(request);}
public DeleteAddressBookResult deleteAddressBook(DeleteAddressBookRequest request) {request = beforeClientExecution(request);return executeDeleteAddressBook(request);}
public void addToolPack(UDFFinder toopack){AggregatingUDFFinder udfs = (AggregatingUDFFinder)_udfFinder;udfs.add(toopack);}
public SearchUsersResult searchUsers(SearchUsersRequest request) {request = beforeClientExecution(request);return executeSearchUsers(request);}
public String getAccessKeySecret() {return privateKeySecret;}
public void setValueAt(int index, E value) {if (mGarbage) {gc();}mValues[index] = value;}
public RefErrorPtg() {field_1_reserved = 0;}
public boolean getFlagByBit(int bitmask) {return ((flags & bitmask) != 0);}
public UpdateAccountSendingEnabledResult updateAccountSendingEnabled(UpdateAccountSendingEnabledRequest request) {request = beforeClientExecution(request);return executeUpdateAccountSendingEnabled(request);}
public AppCookieStickinessPolicy(String policyName, String cookieName) {setPolicyName(policyName);setCookieName(cookieName);}
public GetAccountBalanceResult getAccountBalance(GetAccountBalanceRequest request) {request = beforeClientExecution(request);return executeGetAccountBalance(request);}
public DescribeConversionTasksResult describeConversionTasks(DescribeConversionTasksRequest request) {request = beforeClientExecution(request);return executeDescribeConversionTasks(request);}
public DescribeImagesResult describeImages() {return describeImages(new DescribeImagesRequest());}
public void close() {_closed = true;}
public ListSignalingChannelsResult listSignalingChannels(ListSignalingChannelsRequest request) {request = beforeClientExecution(request);return executeListSignalingChannels(request);}
public MergeFacesRequest() {super("CloudPhoto", "2017-07-11", "MergeFaces", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public DetectTextResult detectText(DetectTextRequest request) {request = beforeClientExecution(request);return executeDetectText(request);}
public DoubleBuffer get(double[] dst) {return get(dst, 0, dst.length);}
public long getCreationTime() {return decodeTS(P_CTIME);}
public TreeFilter clone() {return new Binary(a.clone(), b.clone());}
public ByteBuffer putChar(char value) {int newPosition = position + SizeOf.CHAR;if (newPosition > limit) {throw new BufferOverflowException();}Memory.pokeShort(backingArray, offset + position, (short) value, order);position = newPosition;return this;}
public String toString() {return String.format("Rect [(%d,%d)-(%d,%d): %dx%d]", x, y, x + w, y + h, w, h);}
public static LongBuffer wrap(long[] array) {return wrap(array, 0, array.length);}
public CharsRef clone() {return new CharsRef(chars, offset, length);}
public SpanNearClauseFactory(IndexReader reader, String fieldName, BasicQueryFactory qf) {this.reader = reader;this.fieldName = fieldName;this.weightBySpanQuery = new HashMap<>();this.qf = qf;}
public BeginRecord clone() {return copy();}
public int start() {return start(0);}
public DescribeGameSessionQueuesResult describeGameSessionQueues(DescribeGameSessionQueuesRequest request) {request = beforeClientExecution(request);return executeDescribeGameSessionQueues(request);}
public SubmitAttachmentStateChangesResult submitAttachmentStateChanges(SubmitAttachmentStateChangesRequest request) {request = beforeClientExecution(request);return executeSubmitAttachmentStateChanges(request);}
public UnicodeString getString(int id ){return field_3_strings.get( id );}
public BigInteger getSignificand() {return _significand;}